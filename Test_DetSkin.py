import cv2
import numpy as np
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA = 50000

LOWER_SKIN = np.array([10, 40, 150])
UPPER_SKIN = np.array([35, 255, 255])

KERNEL_SIZE = 5

OUTPUT_IMAGE       = "detected_fake_skin_result.jpg"
OUTPUT_MASK        = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_corners_and_bbox.json"
OUTPUT_POINTS_TXT  = "fake_skin_corners_and_bbox.txt"

OUTPUT_JOB = "SkinDetect.JBI"
JOB_NAME   = "SkinDetect"

COLOR_MODE = "NO_SWAP"

MOVEJ_SPEED = 0.78
MOVL_SPEED  = 20.0

# =========================================================
# ROBOT CALIBRATION POINTS FROM YOUR TATCALIBRATE.JBI
# Order: TL, TR, BR, BL
# =========================================================

ROBOT_WORKSPACE_PULSE = [
    [-43487, 10808, -30891, 5060, -37212, 17750],   # TL = C00001
    [-24167, 57720,  22702, 2808, -52785, 10258],   # TR = C00002
    [ 23516, 61116,  27250,-1344, -54873,-10642],   # BR = C00003
    [ 42600, 11752, -30041,-3542, -38135,-18280],   # BL = C00004
]

# Camera image is assumed to represent the full robot workspace
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

CAMERA_WORKSPACE_PIXELS = np.array([
    [0,            0],
    [CAMERA_WIDTH, 0],
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
# CAMERA
# =========================================================

def start_picamera():
    from picamera2 import Picamera2
    import time

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )

    picam2.configure(config)

    picam2.set_controls({
        "AwbEnable": True,
        "AeEnable": True,
    })

    picam2.start()
    time.sleep(2)

    return picam2


def get_frame(picam2):
    raw = picam2.capture_array()
    return fix_pi_camera_color(raw)

# =========================================================
# POINT ORDERING
# =========================================================

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(diff)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

# =========================================================
# FAKE SKIN DETECTION
# =========================================================

def detect_fake_skin_rotated_box(frame):
    output = frame.copy()

    detection_frame = improve_color_for_detection(frame)
    hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return output, mask, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_AREA:
        return output, mask, None

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)
    ordered_box = order_points(box)

    cv2.drawContours(output, [largest], -1, (255, 0, 0), 2)
    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    labels = ["P1_TL", "P2_TR", "P3_BR", "P4_BL"]

    for label, pt in zip(labels, ordered_box):
        x, y = int(pt[0]), int(pt[1])

        cv2.circle(output, (x, y), 8, (0, 255, 255), -1)

        cv2.putText(
            output,
            f"{label}: ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    p_points = {
        "P1_TL": [int(ordered_box[0][0]), int(ordered_box[0][1])],
        "P2_TR": [int(ordered_box[1][0]), int(ordered_box[1][1])],
        "P3_BR": [int(ordered_box[2][0]), int(ordered_box[2][1])],
        "P4_BL": [int(ordered_box[3][0]), int(ordered_box[3][1])],
    }

    detection_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "center_px": [float(cx), float(cy)],
        "bounding_box": {
            "width_px": float(w),
            "height_px": float(h),
            "angle_deg": float(angle),
            "area_px": float(area),
        },
        "corner_points_px": p_points,
        "box_points_ordered_list": ordered_box.tolist(),
    }

    return output, mask, detection_info

# =========================================================
# CAMERA PIXEL TO NORMALIZED WORKSPACE
# =========================================================

def get_camera_to_normalized_workspace_matrix():
    normalized_workspace = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(
        CAMERA_WORKSPACE_PIXELS,
        normalized_workspace,
    )


def pixel_to_normalized(pixel_point, H):
    src = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)

    u = float(dst[0][0][0])
    v = float(dst[0][0][1])

    # Clamp so the robot never goes outside the taught workspace
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    return u, v

# =========================================================
# NORMALIZED WORKSPACE TO ROBOT PULSE
# =========================================================

def bilinear_pulse_from_normalized(u, v):
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    TL = np.array(ROBOT_WORKSPACE_PULSE[0], dtype=np.float64)
    TR = np.array(ROBOT_WORKSPACE_PULSE[1], dtype=np.float64)
    BR = np.array(ROBOT_WORKSPACE_PULSE[2], dtype=np.float64)
    BL = np.array(ROBOT_WORKSPACE_PULSE[3], dtype=np.float64)

    top = TL + u * (TR - TL)
    bottom = BL + u * (BR - BL)

    point = top + v * (bottom - top)

    return [int(round(x)) for x in point]

# =========================================================
# JBI WRITER
# =========================================================

def write_jbi_corner_visit(robot_pulse_points, filename=OUTPUT_JOB, job_name=JOB_NAME):
    if len(robot_pulse_points) != 4:
        raise ValueError(f"Expected 4 points, got {len(robot_pulse_points)}")

    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    # Start at first detected corner
    first_idx = add_point(robot_pulse_points[0])
    instructions.append(f"MOVJ C{first_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # Trace other corners
    for pulse in robot_pulse_points[1:]:
        idx = add_point(pulse)
        instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    # Close rectangle
    close_idx = add_point(robot_pulse_points[0])
    instructions.append(f"MOVL C{close_idx:05d} V={MOVL_SPEED:.1f}")

    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(all_points)},0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    for i, p in enumerate(all_points):
        lines.append(f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")

    lines.append("//INST")
    lines.append(f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI file: {filename}")

# =========================================================
# SAVE RESULTS
# =========================================================

def save_detection_files(detection_info, robot_pulse_points):
    detection_info["robot_pulse_points"] = {
        "P1_TL": robot_pulse_points[0],
        "P2_TR": robot_pulse_points[1],
        "P3_BR": robot_pulse_points[2],
        "P4_BL": robot_pulse_points[3],
    }

    with open(OUTPUT_POINTS_JSON, "w") as f:
        json.dump(detection_info, f, indent=4)

    with open(OUTPUT_POINTS_TXT, "w") as f:
        f.write("FAKE SKIN DETECTION RESULTS\n")
        f.write("===========================\n\n")
        f.write(f"Timestamp: {detection_info['timestamp']}\n\n")

        f.write("Bounding Box:\n")
        f.write(f"  Width px:  {detection_info['bounding_box']['width_px']:.2f}\n")
        f.write(f"  Height px: {detection_info['bounding_box']['height_px']:.2f}\n")
        f.write(f"  Angle deg: {detection_info['bounding_box']['angle_deg']:.2f}\n")
        f.write(f"  Area px:   {detection_info['bounding_box']['area_px']:.2f}\n\n")

        f.write("Camera Pixel Corners:\n")
        for name, pt in detection_info["corner_points_px"].items():
            f.write(f"  {name}: {pt}\n")

        f.write("\nRobot Pulse Points:\n")
        for name, pulse in detection_info["robot_pulse_points"].items():
            f.write(f"  {name}: {pulse}\n")

    print(f"Saved {OUTPUT_POINTS_JSON}")
    print(f"Saved {OUTPUT_POINTS_TXT}")

# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame):
    annotated, mask, info = detect_fake_skin_rotated_box(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    if info is None:
        print("\nNo fake skin detected.")
        print(f"Saved {OUTPUT_IMAGE}")
        print(f"Saved {OUTPUT_MASK}")
        return

    print("\nDetected fake skin.")

    H = get_camera_to_normalized_workspace_matrix()

    robot_pulse_points = []

    print("\nConverted fake skin corners to Yaskawa pulse positions:")

    for label, pixel_pt in info["corner_points_px"].items():
        u, v = pixel_to_normalized(pixel_pt, H)

        pulse = bilinear_pulse_from_normalized(u, v)
        robot_pulse_points.append(pulse)

        print(f"\n{label}")
        print(f"  pixel: {pixel_pt}")
        print(f"  uv:    u={u:.4f}, v={v:.4f}")
        print(f"  pulse: {pulse}")

    write_jbi_corner_visit(robot_pulse_points)
    save_detection_files(info, robot_pulse_points)

    print("\nSaved files:")
    print(f"  {OUTPUT_IMAGE}")
    print(f"  {OUTPUT_MASK}")
    print(f"  {OUTPUT_POINTS_JSON}")
    print(f"  {OUTPUT_POINTS_TXT}")
    print(f"  {OUTPUT_JOB}")

# =========================================================
# MAIN
# =========================================================

def main():
    print("\n=== Fake Skin Detection → SkinDetect.JBI ===\n")

    picam2 = start_picamera()

    print("Controls:")
    print("  SPACE = capture frame, detect fake skin, generate SkinDetect.JBI")
    print("  Q     = quit")
    print("  1     = RGB_TO_BGR color mode")
    print("  2     = NO_SWAP color mode")
    print("  3     = BGR_TO_RGB color mode\n")

    global COLOR_MODE

    while True:
        frame = get_frame(picam2)
        preview = frame.copy()

        cv2.putText(
            preview,
            f"COLOR_MODE: {COLOR_MODE}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        # Draw camera workspace border
        pts = CAMERA_WORKSPACE_PIXELS.astype(int)
        cv2.polylines(preview, [pts], True, (255, 0, 255), 3)

        for label, pt in zip(["TL", "TR", "BR", "BL"], pts):
            cv2.circle(preview, tuple(pt), 8, (255, 0, 255), -1)
            cv2.putText(
                preview,
                label,
                (pt[0] + 8, pt[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
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

        elif key == ord(" "):
            process_frame(frame.copy())

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
