import cv2
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

COLOR_MODE = "NO_SWAP"
OUTPUT_JSON = "skin_calibration_pulse.json"

ROUNDS = 3
POINTS_PER_ROUND = 4

# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# =========================================================
# COLOR
# =========================================================

def fix_pi_camera_color(frame):
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.copy()

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
# CALIBRATION GLOBALS
# =========================================================

clicked_point = None
LABELS = ["TL", "TR", "BR", "BL"]

# =========================================================
# MOUSE CLICK
# =========================================================

def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"\nCLICKED PIXEL: x={x}, y={y}")

# =========================================================
# PULSE INPUT
# =========================================================

def get_pulse_position(label):
    print("\n=================================================")
    print(f"ENTER PULSE COORDINATES FOR {label}")
    print("=================================================")
    print("Jog the robot needle/tool tip to the SAME physical point.")
    print("Then enter the PULSE values from the pendant:")
    print("S, L, U, R, B, T")
    print()

    pulse_labels = ["S", "L", "U", "R", "B", "T"]

    while True:
        values = {}

        try:
            for item in pulse_labels:
                values[item] = int(float(input(f"{item}: ")))

            return values

        except ValueError:
            print("\nERROR: Type numbers only. Try this point again.\n")

# =========================================================
# DRAW SAVED POINTS
# =========================================================

def draw_saved_points(frame, points):
    for point in points:
        x = point["camera_pixel"]["x"]
        y = point["camera_pixel"]["y"]
        label = point["label"]

        cv2.circle(frame, (x, y), 9, (0, 255, 0), -1)

        cv2.putText(
            frame,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

# =========================================================
# Z INPUT
# =========================================================

def get_specific_z():
    print("\n=================================================")
    print("ENTER SPECIFIC Z HEIGHT")
    print("=================================================")
    print("This is the Z value you want the later robot code to use.")
    print("Rx, Ry, and Rz will be ignored for now.")
    print()

    while True:
        try:
            z = float(input("Specific Z height: "))
            return z
        except ValueError:
            print("\nERROR: Type a number only.\n")

# =========================================================
# SAVE JSON
# =========================================================

def save_json(data):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)

    print("\n=================================================")
    print("PULSE CALIBRATION COMPLETE")
    print("=================================================")
    print(f"Saved calibration file: {OUTPUT_JSON}")
    print("=================================================\n")

# =========================================================
# MAIN
# =========================================================

def main():
    global clicked_point

    calibration_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": "Manual fake skin camera-to-robot pulse calibration",
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "color_mode": COLOR_MODE,
        "coordinate_type": "PULSE",
        "pulse_order": ["S", "L", "U", "R", "B", "T"],
        "ignored_cartesian_rotation": ["Rx", "Ry", "Rz"],
        "round_count": ROUNDS,
        "points_per_round": POINTS_PER_ROUND,
        "point_order": LABELS,
        "rounds": []
    }

    print("\n=================================================")
    print("FAKE SKIN 3-ROUND PULSE CALIBRATION")
    print("=================================================")
    print("For EACH round:")
    print("  1. Place fake skin on the board.")
    print("  2. Click TL, TR, BR, BL in the camera window.")
    print("  3. For each clicked point, enter S, L, U, R, B, T.")
    print("  4. Move the fake skin to a new position.")
    print()
    print("At the end, you will enter ONE specific Z height.")
    print()
    print("Controls:")
    print("  Left Click = select skin corner")
    print("  Q = quit")
    print("=================================================\n")

    picam2 = start_picamera()

    window_name = "Pi Camera Preview"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        for round_num in range(1, ROUNDS + 1):
            print("\n=================================================")
            print(f"ROUND {round_num} OF {ROUNDS}")
            print("=================================================")
            print("Place the fake skin in position for this round.")
            input("Press ENTER when ready...")

            round_points = []

            for point_index, label in enumerate(LABELS, start=1):
                clicked_point = None

                print("\n-------------------------------------------------")
                print(f"ROUND {round_num} - POINT {point_index}: {label}")
                print(f"Click the {label} corner/dot in the camera window.")
                print("Then enter the matching pulse coordinates.")
                print("-------------------------------------------------")

                while clicked_point is None:
                    frame = get_frame(picam2)
                    preview = frame.copy()

                    draw_saved_points(preview, round_points)

                    cv2.putText(
                        preview,
                        f"Round {round_num}/3 - Click {label}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        preview,
                        "Point order: TL, TR, BR, BL",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        preview,
                        "Left click point | Q=quit",
                        (20, CAMERA_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow(window_name, preview)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("\nCalibration cancelled.")
                        return

                camera_x, camera_y = clicked_point

                print(f"\nSaved camera pixel for {label}:")
                print(f"  x = {camera_x}")
                print(f"  y = {camera_y}")

                pulse_position = get_pulse_position(label)

                point_data = {
                    "label": label,
                    "camera_pixel": {
                        "x": int(camera_x),
                        "y": int(camera_y)
                    },
                    "robot_pulse": pulse_position
                }

                round_points.append(point_data)

                print(f"\nSAVED: Round {round_num}, Point {point_index}, {label}")

            calibration_data["rounds"].append({
                "round_number": round_num,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "skin_points": {
                    point["label"]: {
                        "camera_pixel": point["camera_pixel"],
                        "robot_pulse": point["robot_pulse"]
                    }
                    for point in round_points
                },
                "points": round_points
            })

            print("\n=================================================")
            print(f"ROUND {round_num} COMPLETE")
            print("=================================================")

            if round_num < ROUNDS:
                print("Move the fake skin to a NEW position.")
                input("Press ENTER when ready for the next round...")

        calibration_data["specific_z_height"] = get_specific_z()

        save_json(calibration_data)

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
