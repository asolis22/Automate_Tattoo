import cv2
import numpy as np
import os
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA = 8000

LOWER_SKIN = np.array([5, 20, 120])
UPPER_SKIN = np.array([35, 170, 255])

KERNEL_SIZE = 5

OUTPUT_IMAGE = "detected_fake_skin_result.jpg"
OUTPUT_MASK = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_corners_and_bbox.json"
OUTPUT_POINTS_TXT = "fake_skin_corners_and_bbox.txt"
OUTPUT_JOB = "SKINCORN.JBI"

CAMERA_WORKSPACE_FILE = "camera_workspace_pixels.json"

COLOR_MODE = "NO_SWAP"

# =========================================================
# ROBOT CALIBRATION POINTS FROM YOUR CALIBRATE JOB
# Order: TL, TR, BR, BL
# =========================================================

ROBOT_WORKSPACE_PULSE = [
    [-45973, 25480, -34069, 6625, -26560, 17869],   # TL
    [-21177, 96617, 68300, 1964, -66663, 9581],     # TR
    [21513, 99872, 73176, -2006, -68597, -9792],    # BR
    [59395, 7268, -46134, -7854, -27054, -23468],   # BL
]

# =========================================================
# COLOR FIX
# =========================================================

def fix_pi_camera_color(frame):
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return frame.copy()


def improve_color_for_detection(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.05
    hsv[:, :, 2] *= 1.02
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# =========================================================
# CAMERA
# =========================================================

def start_picamera():
    from picamera2 import Picamera2

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
    return picam2


def get_frame(picam2):
    raw = picam2.capture_array()
    frame = fix_pi_camera_color(raw)
    frame = improve_color_for_detection(frame)
    return frame


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
# CLICK CAMERA WORKSPACE CALIBRATION
# =========================================================

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: [{x}, {y}]")


def calibrate_camera_workspace(picam2):
    """
    Click the 4 workspace corners in this order:
    TL, TR, BR, BL
    """
    global clicked_points
    clicked_points = []

    print("\nCAMERA WORKSPACE CALIBRATION")
    print("Click the FULL robot workspace corners in this order:")
    print("1. Top Left")
    print("2. Top Right")
    print("3. Bottom Right")
    print("4. Bottom Left")
    print("Press Q to cancel.\n")

    cv2.namedWindow("Click Workspace Corners: TL, TR, BR, BL")
    cv2.setMouseCallback("Click Workspace Corners: TL, TR, BR, BL", mouse_callback)

    while True:
        frame = get_frame(picam2)
        preview = frame.copy()

        labels = ["TL", "TR", "BR", "BL"]

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
                2
            )

        cv2.putText(
            preview,
            f"Clicked {len(clicked_points)}/4 corners",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow("Click Workspace Corners: TL, TR, BR, BL", preview)

        key = cv2.waitKey(1) & 0xFF

        if len(clicked_points) == 4:
            break

        if key == ord("q"):
            cv2.destroyWindow("Click Workspace Corners: TL, TR, BR, BL")
            return None

    cv2.destroyWindow("Click Workspace Corners: TL, TR, BR, BL")

    workspace_pixels = np.array(clicked_points, dtype=np.float32)

    with open(CAMERA_WORKSPACE_FILE, "w") as f:
        json.dump({
            "order": ["TL", "TR", "BR", "BL"],
            "camera_workspace_pixels": workspace_pixels.astype(int).tolist()
        }, f, indent=4)

    print(f"\nSaved camera workspace calibration to {CAMERA_WORKSPACE_FILE}")
    print(workspace_pixels)

    return workspace_pixels


def load_camera_workspace_pixels():
    if not os.path.exists(CAMERA_WORKSPACE_FILE):
        return None

    with open(CAMERA_WORKSPACE_FILE, "r") as f:
        data = json.load(f)

    return np.array(data["camera_workspace_pixels"], dtype=np.float32)


# =========================================================
# FAKE SKIN DETECTION
# =========================================================

def detect_fake_skin_rotated_box(frame):
    output = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    box = np.int32(box)

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
            2
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
# CAMERA PIXEL TO ROBOT PULSE
# =========================================================

def get_camera_to_normalized_workspace_matrix(camera_workspace_pixels):
    normalized_workspace = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(camera_workspace_pixels, normalized_workspace)


def pixel_to_normalized(pixel_point, H):
    src = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)

    u = float(dst[0][0][0])
    v = float(dst[0][0][1])

    return u, v


def bilinear_pulse_from_normalized(u, v):
    """
    u = 0 left, 1 right
    v = 0 top, 1 bottom
    """

    if u < -0.20 or u > 1.20 or v < -0.20 or v > 1.20:
        raise ValueError(
            f"Detected point is too far outside calibrated workspace: u={u:.3f}, v={v:.3f}"
        )

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
# SAVE FILES
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

        f.write("P1-P4 Camera Pixel Corners:\n")
        for name, pt in detection_info["corner_points_px"].items():
            f.write(f"  {name}: {pt}\n")

        f.write("\nP1-P4 Yaskawa Pulse Points:\n")
        for name, pulse in detection_info["robot_pulse_points"].items():
            f.write(f"  {name}: {pulse}\n")

    print(f"Saved {OUTPUT_POINTS_JSON}")
    print(f"Saved {OUTPUT_POINTS_TXT}")


# =========================================================
# YASKAWA JBI WRITER - BASED ON WORKING FORMAT
# =========================================================

def write_jbi_corner_test(robot_pulse_points, filename=OUTPUT_JOB, job_name="SKINCORN"):
    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    for p in robot_pulse_points:
        if len(p) != 6:
            raise ValueError(f"Bad pulse point: {p}. Each point must have 6 values.")
        add_point(p)

    instructions.append("MOVJ C00000 VJ=0.78")
    instructions.append("MOVL C00001 V=8.0")
    instructions.append("MOVL C00002 V=8.0")
    instructions.append("MOVL C00003 V=8.0")
    instructions.append("MOVL C00000 V=8.0")

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
    lines.append("///DATE 2026/04/22 17:20")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN PROCESS
# =========================================================

def process_frame(frame, camera_workspace_pixels):
    annotated, mask, info = detect_fake_skin_rotated_box(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    if info is None:
        print("\nNo fake skin detected.")
        print(f"Saved {OUTPUT_IMAGE} and {OUTPUT_MASK}")
        return

    print("\nDetected fake skin.")

    print("\nBounding box:")
    print(f"  Width px:  {info['bounding_box']['width_px']:.2f}")
    print(f"  Height px: {info['bounding_box']['height_px']:.2f}")
    print(f"  Angle deg: {info['bounding_box']['angle_deg']:.2f}")
    print(f"  Area px:   {info['bounding_box']['area_px']:.2f}")

    print("\nP1-P4 camera pixel corners:")
    for label, pt in info["corner_points_px"].items():
        print(f"  {label}: {pt}")

    H = get_camera_to_normalized_workspace_matrix(camera_workspace_pixels)

    robot_pulse_points = []

    print("\nConverted fake skin corners to Yaskawa pulse positions:")

    for label, pixel_pt in info["corner_points_px"].items():
        u, v = pixel_to_normalized(pixel_pt, H)

        print(f"{label} normalized: u={u:.4f}, v={v:.4f}")

        pulse = bilinear_pulse_from_normalized(u, v)
        robot_pulse_points.append(pulse)

        print(f"{label} pulse: {pulse}")

    write_jbi_corner_test(robot_pulse_points, filename=OUTPUT_JOB, job_name="SKINCORN")
    save_detection_files(info, robot_pulse_points)

    print("\nSaved files:")
    print(f" - {OUTPUT_IMAGE}")
    print(f" - {OUTPUT_MASK}")
    print(f" - {OUTPUT_POINTS_JSON}")
    print(f" - {OUTPUT_POINTS_TXT}")
    print(f" - {OUTPUT_JOB}")


def main():
    print("\n=== Fake Skin Detection to Yaskawa Corner JBI ===\n")

    picam2 = start_picamera()

    camera_workspace_pixels = load_camera_workspace_pixels()

    print("Controls:")
    print("  C = calibrate/click workspace corners")
    print("  SPACE = capture fake skin and generate JBI")
    print("  Q = quit")
    print("  1 = RGB_TO_BGR")
    print("  2 = NO_SWAP")
    print("  3 = BGR_TO_RGB\n")

    if camera_workspace_pixels is None:
        print("No camera workspace calibration found.")
        print("Press C first and click TL, TR, BR, BL.\n")
    else:
        print(f"Loaded camera workspace calibration from {CAMERA_WORKSPACE_FILE}")

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
            camera_workspace_pixels = calibrate_camera_workspace(picam2)

        elif key == ord(" "):
            if camera_workspace_pixels is None:
                print("\nERROR: You must press C and calibrate the camera workspace first.")
            else:
                captured = frame.copy()
                process_frame(captured, camera_workspace_pixels)

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()