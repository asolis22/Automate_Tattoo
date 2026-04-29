import cv2
import numpy as np
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA    = 30000
MAX_AREA    = 500000
KERNEL_SIZE = 5

LOWER_SKIN = np.array([10, 40, 150])
UPPER_SKIN = np.array([35, 255, 255])

OUTPUT_IMAGE       = "detected_fake_skin_result.jpg"
OUTPUT_MASK        = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_corners_and_bbox.json"
OUTPUT_POINTS_TXT  = "fake_skin_corners_and_bbox.txt"
OUTPUT_JOB         = "SkinDetect.JBI"
JOB_NAME           = "SkinDetect"

COLOR_MODE  = "NO_SWAP"
MOVEJ_SPEED = 0.78
MOVL_SPEED  = 11.0

# =========================================================
# FROM TATCALIBRATE.JBI
# =========================================================

ROBOT_HOME_PULSE = [-7969, 21694, -5134, 1465, -52599, 3149]

# =========================================================
# KNOWN SKIN DIMENSIONS IN PULSE SPACE
# Measured by physically jogging needle to each corner
# while skin was at reference position.
#
# We use these to compute:
#   WIDTH_VEC  = vector from TL to TR
#   HEIGHT_VEC = vector from TL to BL
# These vectors define the skin's real size in robot space.
# =========================================================

REF_TL = np.array([-7415,  19236, -44011,  2716, -21899,  1858], dtype=np.float64)
REF_TR = np.array([20487,  23954, -39126,  -843, -23139, -9100], dtype=np.float64)
REF_BR = np.array([14615,  44219, -13742,  -132, -31300, -6930], dtype=np.float64)
REF_BL = np.array([-5923,  42089, -16782,  1880, -29987,  1548], dtype=np.float64)

# Vectors defining skin size — from TR corner
# TR → TL (left along top edge)
TR_TO_TL = REF_TL - REF_TR

# TR → BR (down along right edge)
TR_TO_BR = REF_BR - REF_TR

# =========================================================
# REFERENCE PIXEL POSITIONS
# Where TR corner appeared in camera when we measured pulses
# =========================================================

REF_TR_PIXEL = np.array([960, 183], dtype=np.float64)
REF_TL_PIXEL = np.array([486, 168], dtype=np.float64)
REF_BR_PIXEL = np.array([949, 541], dtype=np.float64)
REF_BL_PIXEL = np.array([475, 527], dtype=np.float64)

# Reference angle when calibration was done (skin was ~0 deg)
REF_ANGLE_DEG = 0.0

# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# =========================================================
# CALIBRATION MAPPING — single point accurate conversion
# Maps TR pixel → TR pulse accurately using perspective
# transform from all 4 reference pixel↔pulse pairs
# =========================================================

CALIBRATION_POINTS = [
    {"label": "TL", "pixel": [486, 168],
     "pulse": [-7415, 19236, -44011, 2716, -21899, 1858]},
    {"label": "TR", "pixel": [960, 183],
     "pulse": [20487, 23954, -39126, -843, -23139, -9100]},
    {"label": "BR", "pixel": [949, 541],
     "pulse": [14615, 44219, -13742, -132, -31300, -6930]},
    {"label": "BL", "pixel": [475, 527],
     "pulse": [-5923, 42089, -16782, 1880, -29987, 1548]},
]


def pixel_to_pulse_calibrated(pixel_pt):
    pixels = np.array([cp["pixel"] for cp in CALIBRATION_POINTS],
                      dtype=np.float32)
    pulses = np.array([cp["pulse"] for cp in CALIBRATION_POINTS],
                      dtype=np.float64)

    src = pixels
    dst = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    H    = cv2.getPerspectiveTransform(src, dst)
    pt   = np.array([[[float(pixel_pt[0]),
                       float(pixel_pt[1])]]],
                    dtype=np.float32)
    norm = cv2.perspectiveTransform(pt, H)
    u    = float(np.clip(norm[0][0][0], 0.0, 1.0))
    v    = float(np.clip(norm[0][0][1], 0.0, 1.0))

    TL = pulses[0]
    TR = pulses[1]
    BR = pulses[2]
    BL = pulses[3]

    top    = TL + u * (TR - TL)
    bottom = BL + u * (BR - BL)
    result = top + v * (bottom - top)

    return [int(round(x)) for x in result], u, v


# =========================================================
# BUILD FULL BOX FROM TR CORNER + KNOWN SIZE + ROTATION
#
# Since we know the exact skin dimensions in pulse space,
# we only need to accurately find TR, then offset the
# other 3 corners using the known size vectors rotated
# by the detected skin angle.
# =========================================================

def build_pulse_corners_from_tr(tr_pulse, angle_deg):
    tr = np.array(tr_pulse, dtype=np.float64)

    angle_diff = 0.0  # disable rotation for now
    angle_rad  = np.radians(angle_diff)
    cos_a      = np.cos(angle_rad)
    sin_a      = np.sin(angle_rad)

    def rotate_vec(vec):
        rotated    = vec.copy()
        s          = vec[0]
        l          = vec[1]
        rotated[0] = cos_a * s - sin_a * l
        rotated[1] = sin_a * s + cos_a * l
        return rotated

    tr_to_tl_rot = rotate_vec(TR_TO_TL)
    tr_to_br_rot = rotate_vec(TR_TO_BR)

    # TL = TR + TR_TO_TL
    tl = tr + tr_to_tl_rot

    # BR = TR + TR_TO_BR
    br = tr + tr_to_br_rot

    # BL = TL + TR_TO_BR  ← only one vector from TL
    # avoids compounding error of adding two vectors from TR
    bl = tl + tr_to_br_rot

    tr_out = [int(round(x)) for x in tr]
    tl_out = [int(round(x)) for x in tl]
    br_out = [int(round(x)) for x in br]
    bl_out = [int(round(x)) for x in bl]

    return tl_out, tr_out, br_out, bl_out

# =========================================================
# COLOR
# =========================================================

def fix_pi_camera_color(frame):
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.copy()


def improve_color_for_detection(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr,
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.40, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# =========================================================
# CAMERA SETUP
# =========================================================

def start_picamera():
    from picamera2 import Picamera2
    import time
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888",
              "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )
    picam2.configure(config)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    picam2.start()
    time.sleep(2)
    return picam2


def get_frame(picam2):
    raw = picam2.capture_array()
    return fix_pi_camera_color(raw)

# =========================================================
# POINT ORDERING
# =========================================================

def order_box_points(box):
    box      = np.array(box, dtype=np.float32)
    sorted_y = box[np.argsort(box[:, 1])]
    top      = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
    bottom   = sorted_y[2:][np.argsort(sorted_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]],
                    dtype=np.float32)

# =========================================================
# SKIN DETECTION
# =========================================================

def detect_skin(frame):
    output   = frame.copy()
    enhanced = improve_color_for_detection(frame)
    hsv      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask     = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.dilate(mask, kernel, iterations=1)
    mask   = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  No contours found.")
        return output, mask, None

    valid = [c for c in contours
             if MIN_AREA < cv2.contourArea(c) < MAX_AREA]

    if not valid:
        areas = sorted([cv2.contourArea(c) for c in contours],
                       reverse=True)[:5]
        print(f"  No contours in range {MIN_AREA}–{MAX_AREA}.")
        print(f"  Top areas: {[f'{a:.0f}' for a in areas]}")
        return output, mask, None

    largest = max(valid, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    rect              = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect
    box               = cv2.boxPoints(rect)
    ordered_box       = order_box_points(box)

    cv2.drawContours(output, [largest], -1, (255, 50, 0), 2)
    cv2.drawContours(output, [ordered_box.astype(int)], 0,
                     (0, 255, 0), 3)

    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_colors = [
        (0,   255, 255),
        (255, 255,   0),
        (0,     0, 255),
        (255,   0, 255),
    ]

    corners_px = {}
    for lbl, pt, color in zip(corner_labels, ordered_box,
                               corner_colors):
        x, y = int(pt[0]), int(pt[1])
        corners_px[lbl] = [x, y]
        cv2.circle(output, (x, y), 10, color, -1)
        cv2.putText(
            output, f"{lbl}: ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.putText(
        output, f"CENTER ({int(cx)},{int(cy)})",
        (int(cx) + 10, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
    )
    cv2.putText(
        output,
        f"Angle: {angle:.1f} deg   Area: {area:.0f}px   "
        f"Size: {w:.0f}x{h:.0f}px",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    detection_info = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "center_px":  [float(cx), float(cy)],
        "angle_deg":  float(angle),
        "area_px":    float(area),
        "width_px":   float(w),
        "height_px":  float(h),
        "corners_px": corners_px,
    }

    print(f"  Skin detected — area: {area:.0f}px  "
          f"angle: {angle:.1f}°  size: {w:.0f}x{h:.0f}px")
    print(f"  Center: ({cx:.1f}, {cy:.1f})")
    for lbl, pt in corners_px.items():
        print(f"  {lbl}: {pt}")

    return output, mask, detection_info

# =========================================================
# JBI WRITER — MOVL sharp corners
# =========================================================

def write_jbi(robot_pulse_corners,
              filename=OUTPUT_JOB,
              job_name=JOB_NAME):
    # robot_pulse_corners order: TL, TR, BR, BL
    all_points   = []
    instructions = []

    def add_pt(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    # HOME
    h0 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(
        f"MOVJ C{h0:05d} VJ={MOVEJ_SPEED:.2f}")

    # Approach TL with MOVJ
    f0 = add_pt(robot_pulse_corners[0])
    instructions.append(
        f"MOVJ C{f0:05d} VJ={MOVEJ_SPEED:.2f}")

    # Trace TR, BR, BL with MOVL
    for pulse in robot_pulse_corners[1:]:
        idx = add_pt(pulse)
        instructions.append(
            f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    # Close loop back to TL
    c0 = add_pt(robot_pulse_corners[0])
    instructions.append(
        f"MOVL C{c0:05d} V={MOVL_SPEED:.1f}")

    # Return HOME
    h1 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(
        f"MOVJ C{h1:05d} VJ={MOVEJ_SPEED:.2f}")

    lines = [
        "/JOB",
        f"//NAME {job_name}",
        "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0",
        "///POSTYPE PULSE",
        "///PULSE",
    ]
    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}="
            f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}"
        )
    lines += [
        "//INST",
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW",
        "///GROUP1 RB1",
        "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI: {filename}")
    corner_labels = ["TL", "TR", "BR", "BL"]
    for lbl, pulse in zip(corner_labels, robot_pulse_corners):
        print(f"  {lbl}: {pulse}")

# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(detection_info, robot_pulse_corners):
    corner_labels = ["TL", "TR", "BR", "BL"]
    detection_info["robot_pulse_corners"] = {
        lbl: robot_pulse_corners[i]
        for i, lbl in enumerate(corner_labels)
    }
    detection_info["method"] = \
        "TR anchor + known size vectors + rotation"

    with open(OUTPUT_POINTS_JSON, "w") as f:
        json.dump(detection_info, f, indent=4)

    with open(OUTPUT_POINTS_TXT, "w") as f:
        f.write("FAKE SKIN DETECTION RESULTS\n")
        f.write("===========================\n\n")
        f.write(f"Timestamp: {detection_info['timestamp']}\n")
        f.write(
            f"Angle:     {detection_info['angle_deg']:.2f} deg\n")
        f.write(
            f"Area:      {detection_info['area_px']:.0f} px\n\n")
        f.write("Camera Pixel Corners:\n")
        for lbl, pt in detection_info["corners_px"].items():
            f.write(f"  {lbl}: {pt}\n")
        f.write("\nRobot Pulse Corners:\n")
        for lbl, pulse in \
                detection_info["robot_pulse_corners"].items():
            f.write(f"  {lbl}: {pulse}\n")

    print(f"Saved {OUTPUT_POINTS_JSON}")
    print(f"Saved {OUTPUT_POINTS_TXT}")

# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame):
    print("\n" + "="*55)
    print("PROCESSING FRAME")
    print("="*55)

    print("\nStep 1: Detecting skin...")
    annotated, mask, info = detect_skin(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK,  mask)

    if info is None:
        print("\nSkin not detected.")
        print("Press T to open HSV tuner and adjust sliders.")
        return

    # Get TR pixel and detected angle
    tr_pixel  = info["corners_px"]["TR"]
    angle_deg = info["angle_deg"]

    print(f"\nStep 2: Converting TR pixel to pulse...")
    tr_pulse, u, v = pixel_to_pulse_calibrated(tr_pixel)
    print(f"  TR pixel: {tr_pixel} → "
          f"uv=({u:.4f},{v:.4f}) → pulse={tr_pulse}")

    print(f"\nStep 3: Building full box from TR + "
          f"known size (angle={angle_deg:.1f}°)...")
    tl, tr, br, bl = build_pulse_corners_from_tr(
        tr_pulse, angle_deg)

    robot_pulse_corners = [tl, tr, br, bl]
    corner_labels       = ["TL", "TR", "BR", "BL"]
    for lbl, pulse in zip(corner_labels, robot_pulse_corners):
        print(f"  {lbl}: {pulse}")

    print("\nStep 4: Writing SkinDetect.JBI...")
    write_jbi(robot_pulse_corners)

    print("\nStep 5: Saving results...")
    save_results(info, robot_pulse_corners)

    print("\nDone:")
    print(f"  {OUTPUT_IMAGE}")
    print(f"  {OUTPUT_MASK}")
    print(f"  {OUTPUT_POINTS_JSON}")
    print(f"  {OUTPUT_POINTS_TXT}")
    print(f"  {OUTPUT_JOB}")

# =========================================================
# HSV TUNER
# =========================================================

tuner_open = False


def open_hsv_tuner():
    global tuner_open
    tuner_open = True
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H min", "HSV Tuner",
                       int(LOWER_SKIN[0]), 179, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuner",
                       int(UPPER_SKIN[0]), 179, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuner",
                       int(LOWER_SKIN[1]), 255, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuner",
                       int(UPPER_SKIN[1]), 255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuner",
                       int(LOWER_SKIN[2]), 255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuner",
                       int(UPPER_SKIN[2]), 255, lambda x: None)
    print("\nHSV Tuner open.")
    print("Adjust until skin is solid WHITE in mask.")
    print("Press T again to apply and close.")


def read_hsv_tuner():
    return (
        np.array([
            cv2.getTrackbarPos("H min", "HSV Tuner"),
            cv2.getTrackbarPos("S min", "HSV Tuner"),
            cv2.getTrackbarPos("V min", "HSV Tuner"),
        ]),
        np.array([
            cv2.getTrackbarPos("H max", "HSV Tuner"),
            cv2.getTrackbarPos("S max", "HSV Tuner"),
            cv2.getTrackbarPos("V max", "HSV Tuner"),
        ]),
    )

# =========================================================
# MAIN
# =========================================================

def main():
    global LOWER_SKIN, UPPER_SKIN, COLOR_MODE, tuner_open

    print("\n" + "="*55)
    print("  Fake Skin Detection → SkinDetect.JBI")
    print("="*55)
    print("\nMethod: TR anchor + known size vectors + rotation")
    print(f"\nReference TR pulse: {REF_TR.tolist()}")
    print(f"TR→TL vector:       {TR_TO_TL.tolist()}")
    print(f"TR→BR vector:       {TR_TO_BR.tolist()}")

    picam2 = start_picamera()

    print("\nControls:")
    print("  SPACE = detect skin → generate SkinDetect.JBI")
    print("  T     = open/close HSV tuner")
    print("  Q     = quit")
    print("  1/2/3 = color mode\n")

    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()

        if tuner_open:
            lo, hi    = read_hsv_tuner()
            enhanced  = improve_color_for_detection(frame)
            hsv_live  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            live_mask = cv2.inRange(hsv_live, lo, hi)
            kernel    = np.ones((KERNEL_SIZE, KERNEL_SIZE),
                                np.uint8)
            live_mask = cv2.morphologyEx(
                live_mask, cv2.MORPH_OPEN, kernel)
            live_mask = cv2.morphologyEx(
                live_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                live_mask, cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours
                     if MIN_AREA < cv2.contourArea(c) < MAX_AREA]
            if valid:
                largest = max(valid, key=cv2.contourArea)
                rect    = cv2.minAreaRect(largest)
                box     = cv2.boxPoints(rect)
                cv2.drawContours(
                    preview, [box.astype(int)], 0,
                    (0, 255, 0), 3)
                cv2.putText(
                    preview,
                    f"SKIN DETECTED — "
                    f"area: {cv2.contourArea(largest):.0f}px",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    preview,
                    "NO SKIN — adjust HSV sliders",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2,
                )
            cv2.imshow("HSV Mask — skin should be WHITE",
                       live_mask)

        cv2.putText(
            preview, f"COLOR_MODE: {COLOR_MODE}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
        )
        cv2.putText(
            preview,
            f"HSV lo={LOWER_SKIN} hi={UPPER_SKIN}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.circle(
            preview,
            (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2),
            6, (0, 255, 0), -1,
        )
        cv2.putText(
            preview,
            "SPACE=detect | T=HSV tuner | Q=quit",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow("Pi Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            COLOR_MODE = "RGB_TO_BGR"
            print("COLOR_MODE = RGB_TO_BGR")
        elif key == ord("2"):
            COLOR_MODE = "NO_SWAP"
            print("COLOR_MODE = NO_SWAP")
        elif key == ord("3"):
            COLOR_MODE = "BGR_TO_RGB"
            print("COLOR_MODE = BGR_TO_RGB")
        elif key == ord("t") or key == ord("T"):
            if not tuner_open:
                open_hsv_tuner()
            else:
                LOWER_SKIN, UPPER_SKIN = read_hsv_tuner()
                cv2.destroyWindow("HSV Tuner")
                cv2.destroyWindow(
                    "HSV Mask — skin should be WHITE")
                tuner_open = False
                print(f"\nHSV applied:")
                print(f"  LOWER_SKIN = {LOWER_SKIN}")
                print(f"  UPPER_SKIN = {UPPER_SKIN}")
        elif key == ord(" "):
            process_frame(frame.copy())
        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()