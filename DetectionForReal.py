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

COLOR_MODE = "NO_SWAP"

CALIBRATION_JSON = "skin_calibration_pulse.json"

OUTPUT_IMAGE = "detected_fake_skin_result.jpg"
OUTPUT_MASK  = "detected_fake_skin_mask.jpg"
OUTPUT_JSON  = "fake_skin_detected_pulse.json"
OUTPUT_TXT   = "fake_skin_detected_pulse.txt"
OUTPUT_JOB   = "SkinDetect.JBI"
JOB_NAME     = "SkinDetect"

MOVEJ_SPEED = 0.78
MOVL_SPEED  = 11.0

ROBOT_HOME_PULSE = [-7969, 21694, -5134, 1465, -52599, 3149]

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# =========================================================
# CAMERA / COLOR — SAME SETUP
# =========================================================

def fix_pi_camera_color(frame):
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.copy()


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
# LOAD CALIBRATION
# =========================================================

def load_calibration():
    with open(CALIBRATION_JSON, "r") as f:
        data = json.load(f)

    pixel_points = []
    pulse_points = []

    for round_data in data["rounds"]:
        for pt in round_data["points"]:
            px = pt["camera_pixel"]["x"]
            py = pt["camera_pixel"]["y"]

            pulse = pt["robot_pulse"]
            pulse_vec = [
                pulse["S"],
                pulse["L"],
                pulse["U"],
                pulse["R"],
                pulse["B"],
                pulse["T"]
            ]

            pixel_points.append([px, py, 1])
            pulse_points.append(pulse_vec)

    A = np.array(pixel_points, dtype=np.float64)
    B = np.array(pulse_points, dtype=np.float64)

    # Fits pixel x,y to pulse S,L,U,R,B,T
    mapping_matrix, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    specific_z = data.get("specific_z_height", None)

    print("\nLoaded calibration:")
    print(f"  Points used: {len(pixel_points)}")
    print(f"  Specific Z: {specific_z}")

    return mapping_matrix, specific_z


def pixel_to_pulse(pixel, mapping_matrix):
    x, y = pixel
    vec = np.array([x, y, 1], dtype=np.float64)
    pulse = vec @ mapping_matrix
    return [int(round(v)) for v in pulse]

# =========================================================
# SKIN DETECTION
# =========================================================

def improve_color_for_detection(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.40, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def order_box_points(box):
    box = np.array(box, dtype=np.float32)
    sorted_y = box[np.argsort(box[:, 1])]
    top = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
    bottom = sorted_y[2:][np.argsort(sorted_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def detect_skin(frame):
    output = frame.copy()
    enhanced = improve_color_for_detection(frame)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return output, mask, None

    valid = [c for c in contours if MIN_AREA < cv2.contourArea(c) < MAX_AREA]

    if not valid:
        print("No valid fake skin contour found.")
        return output, mask, None

    largest = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)
    ordered_box = order_box_points(box)

    labels = ["TL", "TR", "BR", "BL"]
    corners_px = {}

    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    for label, pt in zip(labels, ordered_box):
        x, y = int(pt[0]), int(pt[1])
        corners_px[label] = [x, y]

        cv2.circle(output, (x, y), 10, (0, 255, 255), -1)
        cv2.putText(
            output,
            f"{label}: ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "center_px": [float(cx), float(cy)],
        "angle_deg": float(angle),
        "area_px": float(area),
        "width_px": float(w),
        "height_px": float(h),
        "corners_px": corners_px
    }

    print("\nSkin detected:")
    print(f"  Area: {area:.0f}")
    print(f"  Center: ({cx:.1f}, {cy:.1f})")
    for label, point in corners_px.items():
        print(f"  {label}: {point}")

    return output, mask, info

# =========================================================
# JBI WRITER
# =========================================================

def write_jbi(robot_pulse_corners):
    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    h0 = add_point(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{h0:05d} VJ={MOVEJ_SPEED:.2f}")

    tl = add_point(robot_pulse_corners["TL"])
    instructions.append(f"MOVJ C{tl:05d} VJ={MOVEJ_SPEED:.2f}")

    for label in ["TR", "BR", "BL", "TL"]:
        idx = add_point(robot_pulse_corners[label])
        instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    h1 = add_point(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{h1:05d} VJ={MOVEJ_SPEED:.2f}")

    lines = [
        "/JOB",
        f"//NAME {JOB_NAME}",
        "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0",
        "///POSTYPE PULSE",
        "///PULSE",
    ]

    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}"
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

    with open(OUTPUT_JOB, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI: {OUTPUT_JOB}")

# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(info, robot_pulse_corners, specific_z):
    info["robot_pulse_corners"] = robot_pulse_corners
    info["specific_z_height"] = specific_z
    info["method"] = "3-round pulse calibration least-squares pixel-to-pulse mapping"

    with open(OUTPUT_JSON, "w") as f:
        json.dump(info, f, indent=4)

    with open(OUTPUT_TXT, "w") as f:
        f.write("FAKE SKIN DETECTION WITH PULSE CALIBRATION\n")
        f.write("=========================================\n\n")
        f.write(f"Timestamp: {info['timestamp']}\n")
        f.write(f"Specific Z height: {specific_z}\n\n")

        f.write("Camera Pixel Corners:\n")
        for label, pt in info["corners_px"].items():
            f.write(f"  {label}: {pt}\n")

        f.write("\nRobot Pulse Corners:\n")
        for label, pulse in robot_pulse_corners.items():
            f.write(f"  {label}: {pulse}\n")

    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_TXT}")

# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame, mapping_matrix, specific_z):
    annotated, mask, info = detect_skin(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    if info is None:
        print("\nSkin not detected.")
        return

    robot_pulse_corners = {}

    for label, pixel in info["corners_px"].items():
        robot_pulse_corners[label] = pixel_to_pulse(pixel, mapping_matrix)

    print("\nMapped robot pulse corners:")
    for label, pulse in robot_pulse_corners.items():
        print(f"  {label}: {pulse}")

    write_jbi(robot_pulse_corners)
    save_results(info, robot_pulse_corners, specific_z)

    print("\nDone.")

# =========================================================
# MAIN
# =========================================================

def main():
    mapping_matrix, specific_z = load_calibration()

    picam2 = start_picamera()

    print("\nControls:")
    print("  SPACE = detect fake skin and create SkinDetect.JBI")
    print("  Q     = quit\n")

    try:
        while True:
            frame = get_frame(picam2)
            preview = frame.copy()

            cv2.putText(
                preview,
                "SPACE=detect skin | Q=quit",
                (20, CAMERA_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Pi Camera Preview", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                process_frame(frame.copy(), mapping_matrix, specific_z)

            elif key == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
