import cv2
import numpy as np
import json
import time
from picamera2 import Picamera2

# ==============================
# CLICK ORDER:
#
# P1  P2  P3
# P4  P5  P6
# P7  P8  P9
# ==============================

clicked_points = []
robot_pulse_points = []

point_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def enter_pulse_values(label):
    print("\n=================================")
    print(f"ENTER PULSE VALUES FOR {label}")
    print("=================================")
    print("Type the robot pulse coordinates from the pendant:")
    print("S, L, U, R, B, T\n")

    while True:
        try:
            s = int(float(input("S: ")))
            l = int(float(input("L: ")))
            u = int(float(input("U: ")))
            r = int(float(input("R: ")))
            b = int(float(input("B: ")))
            t = int(float(input("T: ")))

            return [s, l, u, r, b, t]

        except ValueError:
            print("\n[ERROR] Please enter numbers only. Try again.\n")


def mouse_callback(event, x, y, flags, param):
    global clicked_points, robot_pulse_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 9:
            label = point_labels[len(clicked_points)]

            clicked_points.append([x, y])
            print(f"\n{label} clicked at pixel: ({x}, {y})")

            pulse_values = enter_pulse_values(label)
            robot_pulse_points.append(pulse_values)

            print(f"[OK] {label} saved:")
            print(f"  Pixel: ({x}, {y})")
            print(f"  Pulse: {pulse_values}")

        if len(clicked_points) == 9:
            print("\nAll 9 points clicked and pulse values entered.")
            print("Press S to save calibration.")


def draw_points(frame):
    display = frame.copy()

    for i, pt in enumerate(clicked_points):
        x, y = pt

        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            display,
            point_labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    lines = [
        "9 Point Pulse Calibration",
        "Click points in order:",
        "P1 P2 P3",
        "P4 P5 P6",
        "P7 P8 P9",
        "After each click, enter S L U R B T",
        "S = save | R = reset | Q = quit"
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        y += 30

    return display


def save_calibration():
    if len(clicked_points) != 9:
        print("[ERROR] You must click all 9 points first.")
        return

    if len(robot_pulse_points) != 9:
        print("[ERROR] You must enter pulse values for all 9 points.")
        return

    image_points = np.array(clicked_points, dtype=np.float64)
    pulse_points = np.array(robot_pulse_points, dtype=np.float64)

    # Linear least-squares map:
    # [pixel_x, pixel_y, 1] -> [S, L, U, R, B, T]
    A = np.column_stack([
        image_points[:, 0],
        image_points[:, 1],
        np.ones(len(image_points))
    ])

    mapping_matrix, residuals, rank, singular_values = np.linalg.lstsq(
        A,
        pulse_points,
        rcond=None
    )

    predicted_pulses = A @ mapping_matrix
    errors = np.linalg.norm(predicted_pulses - pulse_points, axis=1)
    avg_error = float(np.mean(errors))

    calibration_data = {
        "description": "Pixel to robot pulse calibration using 9 manually entered pulse points",
        "coordinate_type": "PULSE",
        "pulse_order": ["S", "L", "U", "R", "B", "T"],
        "click_order": point_labels,
        "image_points_pixels": image_points.tolist(),
        "robot_points_pulse": pulse_points.astype(int).tolist(),
        "pixel_to_pulse_mapping_matrix": mapping_matrix.tolist(),
        "calibration_test": {
            "predicted_pulses": predicted_pulses.tolist(),
            "errors": errors.tolist(),
            "average_error": avg_error
        }
    }

    with open("calibration_pulse.json", "w") as f:
        json.dump(calibration_data, f, indent=4)

    print("\n=================================")
    print("PULSE CALIBRATION SAVED")
    print("=================================")
    print("Saved as: calibration_pulse.json")

    print("\nCalibration test:")
    for i in range(9):
        expected = pulse_points[i]
        predicted = predicted_pulses[i]
        error = errors[i]

        print(
            f"{point_labels[i]} | "
            f"Expected={expected.astype(int).tolist()} | "
            f"Predicted={[int(round(v)) for v in predicted]} | "
            f"Error={error:.3f}"
        )

    print(f"\nAverage pulse error: {avg_error:.3f}")


def main():
    global clicked_points, robot_pulse_points

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={
            "size": (1280, 720),
            "format": "RGB888"
        }
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    window_name = "9 Point Robot Pulse Calibration"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=================================")
    print("9 POINT ROBOT PULSE CALIBRATION")
    print("=================================")
    print("Click points in this order:")
    print("P1 P2 P3")
    print("P4 P5 P6")
    print("P7 P8 P9")
    print()
    print("After every click, enter:")
    print("S, L, U, R, B, T")
    print()
    print("Controls:")
    print("S = save calibration_pulse.json")
    print("R = reset clicked points")
    print("Q = quit")
    print("=================================\n")

    while True:
        frame = picam2.capture_array()

        # Picamera2 gives RGB.
        # OpenCV displays BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display = draw_points(frame_bgr)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break

        elif key == ord("r"):
            clicked_points = []
            robot_pulse_points = []
            print("[INFO] Reset clicked points and pulse values. Start again from P1.")

        elif key == ord("s"):
            save_calibration()

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()