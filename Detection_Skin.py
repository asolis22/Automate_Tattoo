import cv2
import numpy as np
import os
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA = 8000

# HSV thresholds for peach/fake skin on blue background
LOWER_SKIN = np.array([5, 20, 120])
UPPER_SKIN = np.array([35, 170, 255])

KERNEL_SIZE = 5

OUTPUT_IMAGE = "detected_fake_skin_result.jpg"
OUTPUT_MASK = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_corners_and_bbox.json"
OUTPUT_POINTS_TXT = "fake_skin_corners_and_bbox.txt"
OUTPUT_JOB = "SKINCORN.JBI"

# Use "auto", "opencv", or "picamera2"
CAMERA_MODE = "auto"

# =========================================================
# YASKAWA WORKSPACE CORNERS - PULSE POSITIONS
# Order: TL, TR, BR, BL
# =========================================================

ROBOT_WORKSPACE_PULSE = [
    [-45973, 25480, -34069, 6625, -26560, 17869],   # TL workspace corner
    [-21177, 96617, 68300, 1964, -66663, 9581],     # TR workspace corner
    [21513, 99872, 73176, -2006, -68597, -9792],    # BR workspace corner
    [59395, 7268, -46134, -7854, -27054, -23468],   # BL workspace corner
]

# =========================================================
# CAMERA WORKSPACE PIXELS
# Replace these with the actual pixel corners of the FULL robot workspace
# as seen by the camera.
# Order: TL, TR, BR, BL
# =========================================================

CAMERA_WORKSPACE_PIXELS = np.array([
    [100, 100],
    [900, 100],
    [900, 700],
    [100, 700],
], dtype=np.float32)


# =========================================================
# POINT ORDERING
# =========================================================

def order_points(pts):
    """
    Orders 4 points as TL, TR, BR, BL.
    """
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    top_right = pts[np.argmin(diff)]
    bottom_right = pts[np.argmax(s)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


# =========================================================
# FAKE SKIN DETECTION
# =========================================================

def detect_fake_skin_rotated_box(frame):
    """
    Detects fake skin and returns:
    - annotated image
    - mask
    - detection_info with bounding box, P1-P4 corners, center, width, height, angle
    """

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

    # Draw contour and rotated bounding box
    cv2.drawContours(output, [largest], -1, (255, 0, 0), 2)
    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    # Label as P1-P4
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

    # Center point
    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.putText(
        output,
        f"Center: ({int(cx)}, {int(cy)})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Bounding Box W:{w:.1f} H:{h:.1f} Angle:{angle:.2f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

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
            "area_px": float(area)
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

    return cv2.getPerspectiveTransform(CAMERA_WORKSPACE_PIXELS, normalized_workspace)


def pixel_to_normalized(pixel_point, H):
    src = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)

    u = float(dst[0][0][0])
    v = float(dst[0][0][1])

    return u, v


# =========================================================
# NORMALIZED WORKSPACE TO YASKAWA PULSE
# =========================================================

def bilinear_pulse_from_normalized(u, v):
    TL = np.array(ROBOT_WORKSPACE_PULSE[0], dtype=np.float64)
    TR = np.array(ROBOT_WORKSPACE_PULSE[1], dtype=np.float64)
    BR = np.array(ROBOT_WORKSPACE_PULSE[2], dtype=np.float64)
    BL = np.array(ROBOT_WORKSPACE_PULSE[3], dtype=np.float64)

    top = TL + u * (TR - TL)
    bottom = BL + u * (BR - BL)
    point = top + v * (bottom - top)

    return [int(round(x)) for x in point]


# =========================================================
# SAVE DETECTION FILES
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
# YASKAWA JBI GENERATOR - BIGSLOW/XYTRACE STYLE
# =========================================================

def generate_yaskawa_jbi(robot_pulse_points, output_path=OUTPUT_JOB):
    job_name = os.path.splitext(os.path.basename(output_path))[0].upper()[:8]

    lines = []

    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(robot_pulse_points)},0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    for i, pulse in enumerate(robot_pulse_points):
        pulse = [int(v) for v in pulse]
        lines.append(f"C{i:05d}=" + ",".join(str(v) for v in pulse))

    lines.append("//INST")
    lines.append("///DATE 2026/04/08 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")

    lines.append("MOVJ C00000 VJ=2.00")

    for i in range(1, len(robot_pulse_points)):
        lines.append(f"MOVL C{i:05d} V=20.0")

    lines.append("MOVL C00000 V=20.0")
    lines.append("END")

    with open(output_path, "w", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved Yaskawa JBI file: {output_path}")


# =========================================================
# CAMERA INPUT
# =========================================================

def capture_with_opencv():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    print("\nOpenCV camera opened.")
    print("Press SPACE to capture.")
    print("Press Q to quit.\n")

    captured_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        cv2.imshow("Camera Preview - SPACE to Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            captured_frame = frame.copy()
            print("Image captured.")
            break

        elif key == ord("q"):
            print("Canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame


def capture_with_picamera2():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("Picamera2 is not installed.")
        return None

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    print("\nPicamera2 opened.")
    print("Press SPACE to capture.")
    print("Press Q to quit.\n")

    captured_frame = None

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("Pi Camera Preview - SPACE to Capture", frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            captured_frame = frame_bgr.copy()
            print("Image captured.")
            break

        elif key == ord("q"):
            print("Canceled.")
            break

    picam2.stop()
    cv2.destroyAllWindows()

    return captured_frame


def capture_image_from_camera():
    if CAMERA_MODE == "picamera2":
        return capture_with_picamera2()

    if CAMERA_MODE == "opencv":
        return capture_with_opencv()

    # auto mode: try Pi Camera first, then OpenCV
    frame = capture_with_picamera2()
    if frame is not None:
        return frame

    print("Trying OpenCV camera instead...")
    return capture_with_opencv()


def load_image_from_file():
    image_path = input("Enter image filename or full path: ").strip()

    if not os.path.exists(image_path):
        print("ERROR: File not found.")
        return None

    frame = cv2.imread(image_path)

    if frame is None:
        print("ERROR: Could not load image.")
        return None

    return frame


# =========================================================
# MAIN PROCESS
# =========================================================

def process_frame(frame):
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

    H = get_camera_to_normalized_workspace_matrix()

    robot_pulse_points = []

    print("\nConverted fake skin corners to Yaskawa pulse positions:")

    for label, pixel_pt in info["corner_points_px"].items():
        u, v = pixel_to_normalized(pixel_pt, H)

        print(f"{label} normalized: u={u:.4f}, v={v:.4f}")

        pulse = bilinear_pulse_from_normalized(u, v)
        robot_pulse_points.append(pulse)

        print(f"{label} pulse: {pulse}")

    generate_yaskawa_jbi(robot_pulse_points, OUTPUT_JOB)
    save_detection_files(info, robot_pulse_points)

    print("\nSaved files:")
    print(f" - {OUTPUT_IMAGE}")
    print(f" - {OUTPUT_MASK}")
    print(f" - {OUTPUT_POINTS_JSON}")
    print(f" - {OUTPUT_POINTS_TXT}")
    print(f" - {OUTPUT_JOB}")

    print("\nIMPORTANT:")
    print("Run the generated JBI in TEACH mode first.")
    print("Keep speed low.")
    print("Be ready to stop the robot if it moves wrong.")


def main():
    print("\n=== Fake Skin Detection to Yaskawa Corner JBI ===\n")
    print("Choose input mode:")
    print("1 = Take picture from camera")
    print("2 = Use saved image file")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        frame = capture_image_from_camera()
    elif choice == "2":
        frame = load_image_from_file()
    else:
        print("Invalid choice.")
        return

    if frame is None:
        print("No image available.")
        return

    process_frame(frame)


if __name__ == "__main__":
    main()
