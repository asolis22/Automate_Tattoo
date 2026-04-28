import cv2
import numpy as np
import os
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
OUTPUT_JOB         = "SKINCORN.JBI"
JOB_NAME           = "SKINCORN"

CAMERA_WORKSPACE_FILE = "camera_workspace_pixels.json"

COLOR_MODE = "NO_SWAP"

# IMPORTANT:
# Measure the real physical width of the calibrated camera workspace.
# This should be the left-to-right physical size of the square the camera sees.
WORKSPACE_WIDTH_CM = 60.0

# Camera-to-needle physical offset
CAM_TO_NEEDLE_LEFT_CM = 16.5
CAM_TO_NEEDLE_DOWN_CM = 38.5
NEEDLE_TO_SKIN_CM     = 39.5

MOVEJ_SPEED = 0.78
MOVL_SPEED  = 20.0

# =========================================================
# ROBOT CALIBRATION POINTS
# Order: TL, TR, BR, BL
# =========================================================

ROBOT_WORKSPACE_PULSE = [
    [-43487, 10808, -30891, 5060, -37212, 17750],   # TL C00001
    [-24167, 57720,  22702, 2808, -52785, 10258],   # TR C00002
    [ 23516, 61116,  27250,-1344, -54873,-10642],   # BR C00003
    [ 42600, 11752, -30041,-3542, -38135,-18280],   # BL C00004
]

CAMERA_WORKSPACE_PIXELS = np.array([
    [0,    0],
    [1280, 0],
    [1280, 720],
    [0,    720],
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
        main={"format": "RGB888", "size": (1280, 720)}
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
# CAMERA WORKSPACE CALIBRATION
# =========================================================

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: [{x}, {y}]")


def calibrate_camera_workspace(picam2):
    global clicked_points
    clicked_points = []

    print("\nCAMERA WORKSPACE CALIBRATION")
    print("Click corners in this order:")
    print("1. Top Left")
    print("2. Top Right")
    print("3. Bottom Right")
    print("4. Bottom Left")
    print("Press Q to cancel.\n")

    window_name = "Click Workspace Corners: TL, TR, BR, BL"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    labels = ["TL", "TR", "BR", "BL"]

    while True:
        frame = get_frame(picam2)
        preview = frame.copy()

        for i, pt in enumerate(clicked_points):
            x, y = pt
            cv2.circle(preview, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(
                preview,
                labels[i],
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            preview,
            f"Clicked {len(clicked_points)}/4 corners",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF

        if len(clicked_points) == 4:
            break

        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)

    workspace_pixels = order_points(clicked_points)

    with open(CAMERA_WORKSPACE_FILE, "w") as f:
        json.dump(
            {
                "order": ["TL", "TR", "BR", "BL"],
                "camera_workspace_pixels": workspace_pixels.astype(int).tolist(),
            },
            f,
            indent=4,
        )

    print(f"\nSaved workspace calibration to {CAMERA_WORKSPACE_FILE}")
    print(workspace_pixels)

    return workspace_pixels


def load_camera_workspace_pixels():
    if os.path.exists(CAMERA_WORKSPACE_FILE):
        with open(CAMERA_WORKSPACE_FILE, "r") as f:
            data = json.load(f)

        pts = np.array(data["camera_workspace_pixels"], dtype=np.float32)
        pts = order_points(pts)

        print(f"Loaded saved camera workspace from {CAMERA_WORKSPACE_FILE}:")
        print(pts)

        return pts

    print("No saved camera workspace found. Using full frame fallback.")
    return CAMERA_WORKSPACE_PIXELS.copy()

# =========================================================
# WORKSPACE MASK
# =========================================================

def keep_only_workspace(frame, camera_workspace_pixels):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = camera_workspace_pixels.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    workspace_only = cv2.bitwise_and(frame, frame, mask=mask)
    return workspace_only, mask

# =========================================================
# FAKE SKIN DETECTION
# =========================================================

def detect_fake_skin_rotated_box(frame, camera_workspace_pixels):
    output = frame.copy()

    workspace_frame, workspace_mask = keep_only_workspace(frame, camera_workspace_pixels)

    detection_frame = improve_color_for_detection(workspace_frame)
    hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    mask = cv2.bitwise_and(mask, mask, mask=workspace_mask)

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

def get_camera_to_normalized_workspace_matrix(camera_workspace_pixels):
    normalized_workspace = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(
        camera_workspace_pixels.astype(np.float32),
        normalized_workspace,
    )


def pixel_to_normalized(pixel_point, H):
    src = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)

    u = float(dst[0][0][0])
    v = float(dst[0][0][1])

    return u, v


def apply_camera_to_needle_offset(u, v):
    """
    The camera sees the point, but the tattoo needle is physically offset.

    Needle is 16.5 cm LEFT of camera.
    In normalized workspace:
        left means u decreases.
    So to place the needle on the detected camera point,
    we shift the robot target in the opposite direction.

    Because the needle is LEFT of the camera, the robot/camera center
    must move RIGHT so the needle lands on the detected point.

    Therefore:
        corrected_u = u + offset_fraction
    """

    offset_u = CAM_TO_NEEDLE_LEFT_CM / WORKSPACE_WIDTH_CM

    corrected_u = u + offset_u
    corrected_v = v

    corrected_u = max(0.0, min(1.0, corrected_u))
    corrected_v = max(0.0, min(1.0, corrected_v))

    return corrected_u, corrected_v

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

    for i, p in enumerate(robot_pulse_points):
        if len(p) != 6:
            raise ValueError(f"Point {i} must have 6 pulse values.")

    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    tl_idx = add_point(robot_pulse_points[0])
    instructions.append(f"MOVJ C{tl_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    for pulse in robot_pulse_points[1:]:
        idx = add_point(pulse)
        instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

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

    print(f"Saved JBI file: {filename}")

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

def process_frame(frame, camera_workspace_pixels):
    annotated, mask, info = detect_fake_skin_rotated_box(frame, camera_workspace_pixels)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    if info is None:
        print("\nNo fake skin detected inside calibrated workspace.")
        return

    print("\nDetected fake skin inside calibrated workspace.")

    H = get_camera_to_normalized_workspace_matrix(camera_workspace_pixels)

    robot_pulse_points = []

    print("\nConverted fake skin corners to Yaskawa pulse positions:")

    for label, pixel_pt in info["corner_points_px"].items():
        u, v = pixel_to_normalized(pixel_pt, H)

        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        corrected_u, corrected_v = apply_camera_to_needle_offset(u, v)

        print(f"\n{label}")
        print(f"  pixel:      {pixel_pt}")
        print(f"  camera uv:  u={u:.4f}, v={v:.4f}")
        print(f"  needle uv:  u={corrected_u:.4f}, v={corrected_v:.4f}")

        pulse = bilinear_pulse_from_normalized(corrected_u, corrected_v)

        robot_pulse_points.append(pulse)

        print(f"  pulse:      {pulse}")

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
    print("\n=== Fake Skin Detection → Yaskawa Corner JBI ===\n")

    picam2 = start_picamera()

    camera_workspace_pixels = load_camera_workspace_pixels()

    print("Controls:")
    print("  C     = calibrate/click workspace corners")
    print("  SPACE = capture frame, detect skin, generate JBI")
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

        if camera_workspace_pixels is not None:
            pts = camera_workspace_pixels.astype(int)
            cv2.polylines(preview, [pts], True, (255, 0, 255), 3)

            for label, pt in zip(["TL", "TR", "BR", "BL"], pts):
                cv2.circle(preview, tuple(pt), 8, (255, 0, 255), -1)
                cv2.putText(
                    preview,
                    label,
                    (pt[0] + 8, pt[1] - 8),
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

        elif key == ord("c"):
            new_workspace = calibrate_camera_workspace(picam2)
            if new_workspace is not None:
                camera_workspace_pixels = new_workspace

        elif key == ord(" "):
            process_frame(frame.copy(), camera_workspace_pixels)

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
