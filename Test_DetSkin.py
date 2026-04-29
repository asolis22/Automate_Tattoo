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
MOVC_SPEED  = 11.0

# =========================================================
# FROM TATCALIBRATE.JBI
# C00000 = home
# C00001=TL, C00002=TR, C00003=BR, C00004=BL
# These 4 corners = full camera view = full workspace
# =========================================================

ROBOT_HOME_PULSE = [-7969, 21694, -5134, 1465, -52599, 3149]

ROBOT_WORKSPACE_PULSE = [
    [-43487, 10808, -30891,  5060, -37212,  17750],  # TL C00001
    [-24167, 57720,  22702,  2808, -52785,  10258],  # TR C00002
    [ 23516, 61116,  27250, -1344, -54873, -10642],  # BR C00003
    [ 42600, 11752, -30041, -3542, -38135, -18280],  # BL C00004
]

# =========================================================
# FROM TATFINDFAKESKIN.JBI
# The 4 exact corners of the fake skin when it was at
# its reference position. We use these to know the skin's
# real-world size and shape in pulse space.
# C00000=TL, C00001=BL, C00002=BR, C00003=TR
# =========================================================

FAKE_SKIN_PULSE_REFERENCE = [
    [ -9288, 47706,  -9166,  2080, -33749,  3204],  # C00000
    [-12327, 27144, -34873,  3063, -25468,  3993],  # C00001
    [ 13267, 27638, -34386,  -733, -25776, -5874],  # C00002
    [  9837, 47750,  -8498,  -174, -34431, -4586],  # C00003
]

# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

CAMERA_WORKSPACE_PIXELS = np.array([
    [0,            0            ],
    [CAMERA_WIDTH, 0            ],
    [CAMERA_WIDTH, CAMERA_HEIGHT],
    [0,            CAMERA_HEIGHT],
], dtype=np.float32)

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
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
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
        main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
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
# COORDINATE PIPELINE
# =========================================================

def get_homography():
    normalized = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(CAMERA_WORKSPACE_PIXELS, normalized)


def pixel_to_normalized(pixel_pt, H):
    src = np.array([[[float(pixel_pt[0]), float(pixel_pt[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    u = max(0.0, min(1.0, float(dst[0][0][0])))
    v = max(0.0, min(1.0, float(dst[0][0][1])))
    return u, v


def bilinear_pulse(u, v):
    TL = np.array(ROBOT_WORKSPACE_PULSE[0], dtype=np.float64)
    TR = np.array(ROBOT_WORKSPACE_PULSE[1], dtype=np.float64)
    BR = np.array(ROBOT_WORKSPACE_PULSE[2], dtype=np.float64)
    BL = np.array(ROBOT_WORKSPACE_PULSE[3], dtype=np.float64)
    top    = TL + u * (TR - TL)
    bottom = BL + u * (BR - BL)
    point  = top + v * (bottom - top)
    return [int(round(x)) for x in point]


def pixel_to_pulse(pixel_pt, H):
    u, v = pixel_to_normalized(pixel_pt, H)
    return bilinear_pulse(u, v), u, v

# =========================================================
# COMPUTE SKIN PULSE CORNERS FROM DETECTED BOX
#
# How this works:
# 1. Camera detects the skin box — gives us pixel corners
# 2. Each pixel corner maps to a robot pulse via bilinear
# 3. BUT we also know the exact shape from TATFINDFAKESKIN
#    so we use the reference to compute the skin's width
#    and height in pulse units, then apply that to the
#    detected center position at the detected angle.
#
# This gives us accurate pulse corners even when the skin
# has moved or rotated to a new position.
# =========================================================

def get_reference_skin_center_pulse():
    """Get the center of the skin in pulse space from reference."""
    pts = np.array(FAKE_SKIN_PULSE_REFERENCE, dtype=np.float64)
    return pts.mean(axis=0)


def get_reference_skin_size_pulse():
    """
    Get width and height of skin in pulse space.
    Uses C00000(TL)→C00003(TR) for width
    and  C00000(TL)→C00001(BL) for height.
    """
    TL = np.array(FAKE_SKIN_PULSE_REFERENCE[0], dtype=np.float64)
    TR = np.array(FAKE_SKIN_PULSE_REFERENCE[3], dtype=np.float64)
    BL = np.array(FAKE_SKIN_PULSE_REFERENCE[1], dtype=np.float64)

    width_vec  = TR - TL   # vector from TL to TR
    height_vec = BL - TL   # vector from TL to BL

    return width_vec, height_vec


def compute_pulse_corners_from_detected_box(detected_corners_px, H):
    """
    Given the 4 pixel corners detected by the camera [TL,TR,BR,BL],
    compute the robot pulse corners by:
    1. Mapping each pixel corner → pulse via bilinear
    2. Computing the detected center in pulse space
    3. Using the reference skin size vectors scaled to match
       the detected pixel size, rotated to match detected angle
    Returns list of 4 pulse points [TL, TR, BR, BL]
    """
    # Convert all 4 detected pixel corners to pulse
    corner_labels = ["TL", "TR", "BR", "BL"]
    detected_pulses = []
    for lbl in corner_labels:
        px = detected_corners_px[lbl]
        pulse, u, v = pixel_to_pulse(px, H)
        detected_pulses.append(np.array(pulse, dtype=np.float64))
        print(f"    {lbl}: pixel={px}  uv=({u:.4f},{v:.4f})  pulse={pulse}")

    # The detected pulse corners are already our answer —
    # the bilinear mapping converts pixel position directly
    # to robot pulse accounting for the workspace geometry.
    # Return them as integer lists.
    return [[int(round(x)) for x in p] for p in detected_pulses]

# =========================================================
# SKIN DETECTION — finds box at any position and rotation
# =========================================================

def order_box_points(box):
    """Order 4 points as TL, TR, BR, BL for any rotation."""
    box = np.array(box, dtype=np.float32)
    sorted_y = box[np.argsort(box[:, 1])]
    top    = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
    bottom = sorted_y[2:][np.argsort(sorted_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def detect_skin(frame):
    """
    Detect the fake skin at any position and rotation.
    Returns annotated frame, mask, detection_info.
    detection_info['corners_px'] = {TL, TR, BR, BL} pixel coords.
    """
    output   = frame.copy()
    enhanced = improve_color_for_detection(frame)
    hsv      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask     = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.dilate(mask, kernel, iterations=1)
    mask   = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  No contours found.")
        return output, mask, None

    valid = [c for c in contours if MIN_AREA < cv2.contourArea(c) < MAX_AREA]

    if not valid:
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)[:5]
        print(f"  No contours in range {MIN_AREA}–{MAX_AREA}.")
        print(f"  Top contour areas found: {[f'{a:.0f}' for a in areas]}")
        return output, mask, None

    largest = max(valid, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Rotated bounding box — handles any angle
    rect              = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect
    box               = cv2.boxPoints(rect)
    ordered_box       = order_box_points(box)

    # Draw contour
    cv2.drawContours(output, [largest], -1, (255, 50, 0), 2)

    # Draw rotated bounding box in green
    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    # Label corners
    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_colors = [
        (0,   255, 255),  # TL cyan
        (255, 255,   0),  # TR yellow
        (0,     0, 255),  # BR red
        (255,   0, 255),  # BL magenta
    ]

    corners_px = {}
    for lbl, pt, color in zip(corner_labels, ordered_box, corner_colors):
        x, y = int(pt[0]), int(pt[1])
        corners_px[lbl] = [x, y]
        cv2.circle(output, (x, y), 10, color, -1)
        cv2.putText(
            output, f"{lbl}: ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    # Center dot
    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.putText(
        output, f"CENTER ({int(cx)},{int(cy)})",
        (int(cx) + 10, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
    )

    # Info text
    cv2.putText(
        output, f"Angle: {angle:.1f} deg   Area: {area:.0f}px   Size: {w:.0f}x{h:.0f}px",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
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

    print(f"  Skin detected — area: {area:.0f}px  angle: {angle:.1f}°  size: {w:.0f}x{h:.0f}px")
    print(f"  Center: ({cx:.1f}, {cy:.1f})")
    for lbl, pt in corners_px.items():
        print(f"  {lbl}: {pt}")

    return output, mask, detection_info

# =========================================================
# JBI WRITER
# =========================================================

def write_jbi(robot_pulse_corners, filename=OUTPUT_JOB, job_name=JOB_NAME):
    all_points   = []
    instructions = []

    def add_pt(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    h0 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{h0:05d} VJ={MOVEJ_SPEED:.2f}")

    f0 = add_pt(robot_pulse_corners[0])
    instructions.append(f"MOVJ C{f0:05d} VJ={MOVEJ_SPEED:.2f}")

    for pulse in robot_pulse_corners:
        idx = add_pt(pulse)
        instructions.append(f"MOVC C{idx:05d} V={MOVC_SPEED:.1f}")

    c0 = add_pt(robot_pulse_corners[0])
    instructions.append(f"MOVC C{c0:05d} V={MOVC_SPEED:.1f}")

    h1 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{h1:05d} VJ={MOVEJ_SPEED:.2f}")

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
        lines.append(f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")
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

    with open(OUTPUT_POINTS_JSON, "w") as f:
        json.dump(detection_info, f, indent=4)

    with open(OUTPUT_POINTS_TXT, "w") as f:
        f.write("FAKE SKIN DETECTION RESULTS\n")
        f.write("===========================\n\n")
        f.write(f"Timestamp: {detection_info['timestamp']}\n")
        f.write(f"Angle:     {detection_info['angle_deg']:.2f} deg\n")
        f.write(f"Area:      {detection_info['area_px']:.0f} px\n\n")
        f.write("Camera Pixel Corners:\n")
        for lbl, pt in detection_info["corners_px"].items():
            f.write(f"  {lbl}: {pt}\n")
        f.write("\nRobot Pulse Corners:\n")
        for lbl, pulse in detection_info["robot_pulse_corners"].items():
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

    print("\nStep 1: Detecting skin position and rotation...")
    annotated, mask, info = detect_skin(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK,  mask)

    if info is None:
        print("\nSkin not detected.")
        print("Tips:")
        print("  — Press T to open HSV tuner")
        print("  — Adjust sliders until skin is solid WHITE in mask window")
        print("  — Press T again to apply")
        return

    print("\nStep 2: Converting pixel corners to robot pulse...")
    H = get_homography()
    robot_pulse_corners = compute_pulse_corners_from_detected_box(
        info["corners_px"], H
    )

    print("\nStep 3: Writing SkinDetect.JBI...")
    write_jbi(robot_pulse_corners)

    print("\nStep 4: Saving results...")
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
    cv2.createTrackbar("H min", "HSV Tuner", int(LOWER_SKIN[0]), 179, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuner", int(UPPER_SKIN[0]), 179, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuner", int(LOWER_SKIN[1]), 255, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuner", int(UPPER_SKIN[1]), 255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuner", int(LOWER_SKIN[2]), 255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuner", int(UPPER_SKIN[2]), 255, lambda x: None)
    print("\nHSV Tuner open.")
    print("Drag sliders until the skin pad is SOLID WHITE in the mask.")
    print("Everything else should be BLACK.")
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
    print("\nWorkspace (TATCALIBRATE.JBI):")
    print(f"  Home: {ROBOT_HOME_PULSE}")
    for lbl, p in zip(["TL","TR","BR","BL"], ROBOT_WORKSPACE_PULSE):
        print(f"  {lbl}: {p}")
    print("\nReference skin size (TATFINDFAKESKIN.JBI):")
    for i, p in enumerate(FAKE_SKIN_PULSE_REFERENCE):
        print(f"  C{i:05d}: {p}")

    picam2 = start_picamera()

    print("\nControls:")
    print("  SPACE = detect skin → generate SkinDetect.JBI")
    print("  T     = open/close HSV tuner")
    print("  Q     = quit")
    print("  1/2/3 = color mode\n")

    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()

        # Live HSV tuner mask preview
        if tuner_open:
            lo, hi = read_hsv_tuner()
            enhanced  = improve_color_for_detection(frame)
            hsv_live  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            live_mask = cv2.inRange(hsv_live, lo, hi)
            kernel    = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
            live_mask = cv2.morphologyEx(live_mask, cv2.MORPH_OPEN,  kernel)
            live_mask = cv2.morphologyEx(live_mask, cv2.MORPH_CLOSE, kernel)

            # Draw live contours on preview so you can see detection live
            contours, _ = cv2.findContours(live_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if MIN_AREA < cv2.contourArea(c) < MAX_AREA]
            if valid:
                largest = max(valid, key=cv2.contourArea)
                rect    = cv2.minAreaRect(largest)
                box     = cv2.boxPoints(rect)
                cv2.drawContours(preview, [box.astype(int)], 0, (0, 255, 0), 3)
                cv2.putText(
                    preview, f"SKIN DETECTED — area: {cv2.contourArea(largest):.0f}px",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    preview, "NO SKIN DETECTED — adjust HSV sliders",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.imshow("HSV Mask — skin should be WHITE", live_mask)

        # HUD
        cv2.putText(
            preview, f"COLOR_MODE: {COLOR_MODE}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
        )
        cv2.putText(
            preview, f"HSV lo={LOWER_SKIN} hi={UPPER_SKIN}",
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # Workspace border
        ws_pts = CAMERA_WORKSPACE_PIXELS.astype(int)
        cv2.polylines(preview, [ws_pts], True, (255, 0, 255), 2)

        # Camera center
        cv2.circle(preview, (CAMERA_WIDTH//2, CAMERA_HEIGHT//2), 6, (0, 255, 0), -1)
        cv2.putText(
            preview, "CAM CENTER",
            (CAMERA_WIDTH//2 + 8, CAMERA_HEIGHT//2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )

        cv2.putText(
            preview, "SPACE=detect | T=HSV tuner | Q=quit",
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
                cv2.destroyWindow("HSV Mask — skin should be WHITE")
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