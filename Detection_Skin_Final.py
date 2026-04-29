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

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

COLOR_MODE = "NO_SWAP"

CALIBRATION_FILE = "calibration.json"

OUTPUT_IMAGE = "detected_fake_skin_result.jpg"
OUTPUT_MASK  = "detected_fake_skin_mask.jpg"
OUTPUT_JSON  = "fake_skin_coordinates_robot.json"

tuner_open = False


# =========================================================
# LOAD CALIBRATION JSON
# =========================================================

def load_calibration():
    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)

        H = np.array(data["homography_pixel_to_robot"], dtype=np.float32)

        print("[OK] Loaded calibration.json")
        print("[OK] Pixel to robot XY calibration ready")

        return H

    except FileNotFoundError:
        print("[ERROR] calibration.json not found.")
        print("Run your 9-point calibration code first.")
        return None

    except KeyError:
        print("[ERROR] calibration.json is missing homography_pixel_to_robot.")
        return None


# =========================================================
# PIXEL TO ROBOT XY
# =========================================================

def pixel_to_robot_xy(pixel_pt, H):
    pixel = np.array(
        [[[float(pixel_pt[0]), float(pixel_pt[1])]]],
        dtype=np.float32
    )

    robot_xy = cv2.perspectiveTransform(pixel, H)[0][0]

    return [float(robot_xy[0]), float(robot_xy[1])]


# =========================================================
# COLOR FUNCTIONS
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
        main={
            "format": "RGB888",
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT)
        }
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
# ORDER BOX POINTS
# =========================================================

def order_box_points(box):
    box = np.array(box, dtype=np.float32)

    sorted_y = box[np.argsort(box[:, 1])]

    top = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
    bottom = sorted_y[2:][np.argsort(sorted_y[2:, 0])]

    # TL, TR, BR, BL
    return np.array([top[0], top[1], bottom[1], bottom[0]],
                    dtype=np.float32)


# =========================================================
# DETECT FAKE SKIN
# =========================================================

def detect_skin(frame, H):
    output = frame.copy()

    enhanced = improve_color_for_detection(frame)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("[ERROR] No contours found.")
        return output, mask, None

    valid = [
        c for c in contours
        if MIN_AREA < cv2.contourArea(c) < MAX_AREA
    ]

    if not valid:
        areas = sorted(
            [cv2.contourArea(c) for c in contours],
            reverse=True
        )[:5]

        print(f"[ERROR] No contours in area range {MIN_AREA}–{MAX_AREA}.")
        print(f"Top contour areas: {[f'{a:.0f}' for a in areas]}")

        return output, mask, None

    largest = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)
    ordered_box = order_box_points(box)

    cv2.drawContours(output, [largest], -1, (255, 50, 0), 2)
    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_colors = [
        (0, 255, 255),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
    ]

    corners_px = {}
    corners_robot_xy = {}

    for lbl, pt, color in zip(corner_labels, ordered_box, corner_colors):
        x, y = int(pt[0]), int(pt[1])

        corners_px[lbl] = [x, y]
        corners_robot_xy[lbl] = pixel_to_robot_xy([x, y], H)

        robot_x, robot_y = corners_robot_xy[lbl]

        cv2.circle(output, (x, y), 10, color, -1)

        cv2.putText(
            output,
            f"{lbl} px=({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        cv2.putText(
            output,
            f"X={robot_x:.1f}, Y={robot_y:.1f}",
            (x + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    center_px = [float(cx), float(cy)]
    center_robot_xy = pixel_to_robot_xy(center_px, H)

    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    cv2.putText(
        output,
        f"CENTER px=({int(cx)},{int(cy)})",
        (int(cx) + 10, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Robot Center X={center_robot_xy[0]:.2f}, Y={center_robot_xy[1]:.2f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Angle: {angle:.1f} deg | Area: {area:.0f}px | Size: {w:.0f}x{h:.0f}px",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    detection_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "HSV fake skin detection + calibration.json homography",
        "center_px": center_px,
        "center_robot_xy_mm": center_robot_xy,
        "angle_deg": float(angle),
        "area_px": float(area),
        "width_px": float(w),
        "height_px": float(h),
        "corners_px": corners_px,
        "corners_robot_xy_mm": corners_robot_xy
    }

    print("\n[OK] Fake skin detected")
    print(f"Area: {area:.0f}px")
    print(f"Angle: {angle:.1f} deg")
    print(f"Center pixel: {center_px}")
    print(f"Center robot XY: X={center_robot_xy[0]:.3f}, Y={center_robot_xy[1]:.3f}")

    print("\nCorners:")
    for lbl in corner_labels:
        px = corners_px[lbl]
        xy = corners_robot_xy[lbl]
        print(f"{lbl}: pixel={px} | robot X={xy[0]:.3f}, Y={xy[1]:.3f}")

    return output, mask, detection_info


# =========================================================
# SAVE JSON RESULTS
# =========================================================

def save_results(info):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n[OK] Saved JSON file: {OUTPUT_JSON}")
    print("JSON contains:")
    print("  - center_px")
    print("  - center_robot_xy_mm")
    print("  - corners_px")
    print("  - corners_robot_xy_mm")
    print("  - angle_deg")
    print("  - area_px")


# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame, H):
    print("\n" + "=" * 55)
    print("PROCESSING FRAME")
    print("=" * 55)

    annotated, mask, info = detect_skin(frame, H)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    print(f"\n[OK] Saved image: {OUTPUT_IMAGE}")
    print(f"[OK] Saved mask: {OUTPUT_MASK}")

    if info is None:
        print("\n[ERROR] Skin not detected.")
        print("Press T to open HSV tuner and adjust sliders.")
        return

    save_results(info)

    print("\nDone.")


# =========================================================
# HSV TUNER
# =========================================================

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

    print("\nHSV Tuner opened.")
    print("Adjust until fake skin is solid WHITE in the mask.")
    print("Press T again to apply and close.")


def read_hsv_tuner():
    lower = np.array([
        cv2.getTrackbarPos("H min", "HSV Tuner"),
        cv2.getTrackbarPos("S min", "HSV Tuner"),
        cv2.getTrackbarPos("V min", "HSV Tuner"),
    ])

    upper = np.array([
        cv2.getTrackbarPos("H max", "HSV Tuner"),
        cv2.getTrackbarPos("S max", "HSV Tuner"),
        cv2.getTrackbarPos("V max", "HSV Tuner"),
    ])

    return lower, upper


# =========================================================
# MAIN
# =========================================================

def main():
    global LOWER_SKIN, UPPER_SKIN, COLOR_MODE, tuner_open

    print("\n" + "=" * 60)
    print(" FAKE SKIN DETECTION WITH ROBOT XY COORDINATES")
    print("=" * 60)
    print("Uses calibration.json")
    print("Does NOT create a JBI file")
    print("Press SPACE to detect and save JSON\n")

    H = load_calibration()

    if H is None:
        print("\nCannot continue without calibration.json.")
        return

    picam2 = start_picamera()

    print("\nControls:")
    print("  SPACE = detect fake skin and save coordinates")
    print("  T     = open/close HSV tuner")
    print("  Q     = quit")
    print("  1     = COLOR_MODE RGB_TO_BGR")
    print("  2     = COLOR_MODE NO_SWAP")
    print("  3     = COLOR_MODE BGR_TO_RGB\n")

    while True:
        frame = get_frame(picam2)
        preview = frame.copy()

        if tuner_open:
            lo, hi = read_hsv_tuner()

            enhanced = improve_color_for_detection(frame)
            hsv_live = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            live_mask = cv2.inRange(hsv_live, lo, hi)

            kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

            live_mask = cv2.morphologyEx(live_mask, cv2.MORPH_OPEN, kernel)
            live_mask = cv2.morphologyEx(live_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                live_mask,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )

            valid = [
                c for c in contours
                if MIN_AREA < cv2.contourArea(c) < MAX_AREA
            ]

            if valid:
                largest = max(valid, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                ordered_box = order_box_points(box)

                cv2.drawContours(preview, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

                cv2.putText(
                    preview,
                    f"SKIN DETECTED area={cv2.contourArea(largest):.0f}px",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    preview,
                    "NO SKIN - adjust HSV sliders",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("HSV Mask - skin should be WHITE", live_mask)

        cv2.putText(
            preview,
            f"COLOR_MODE: {COLOR_MODE}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            preview,
            f"HSV lo={LOWER_SKIN} hi={UPPER_SKIN}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.putText(
            preview,
            "SPACE=detect/save JSON | T=HSV tuner | Q=quit",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
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
                cv2.destroyWindow("HSV Mask - skin should be WHITE")

                tuner_open = False

                print("\nHSV applied:")
                print(f"  LOWER_SKIN = {LOWER_SKIN}")
                print(f"  UPPER_SKIN = {UPPER_SKIN}")

        elif key == ord(" "):
            process_frame(frame.copy(), H)

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
