import cv2
import numpy as np
from picamera2 import Picamera2
import time

# ==============================
# 9 ROBOT CALIBRATION POINTS
# Click these in row order:
#
# P1  P2  P3
# P4  P5  P6
# P7  P8  P9
# ==============================
robot_points = np.array([
    [266.281, -103.412],  # P1
    [264.545,   24.239],  # P2
    [271.393,  154.316],  # P3

    [376.842, -108.441],  # P4
    [375.320,   25.230],  # P5
    [379.985,  160.264],  # P6

    [488.746, -111.483],  # P7
    [488.185,   25.841],  # P8
    [490.716,  164.271],  # P9
], dtype=np.float32)

clicked_points = []
point_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 9:
            clicked_points.append([x, y])
            label = point_labels[len(clicked_points) - 1]
            print(f"{label} clicked at pixel: ({x}, {y})")

        if len(clicked_points) == 9:
            print("\nAll 9 points clicked.")
            print("Press S to save calibration.")


def draw_points(frame):
    display = frame.copy()

    for i, pt in enumerate(clicked_points):
        x, y = pt
        cv2.circle(display, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(
            display,
            point_labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    instructions = [
        "Click points in this order:",
        "P1 P2 P3",
        "P4 P5 P6",
        "P7 P8 P9",
        "S = save | R = reset | Q = quit"
    ]

    y = 30
    for line in instructions:
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


def save_homography():
    if len(clicked_points) != 9:
        print("[ERROR] Click all 9 points first.")
        return

    image_points = np.array(clicked_points, dtype=np.float32)

    H, mask = cv2.findHomography(
        image_points,
        robot_points,
        cv2.RANSAC
    )

    if H is None:
        print("[ERROR] Homography failed.")
        return

    np.save("homography_pixel_to_robot.npy", H)

    print("\n=================================")
    print("CALIBRATION SAVED")
    print("=================================")
    print("Saved as: homography_pixel_to_robot.npy")
    print("\nHomography matrix:")
    print(H)

    print("\nCalibration test:")
    for i, pt in enumerate(image_points):
        pixel = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        predicted = cv2.perspectiveTransform(pixel, H)[0][0]
        expected = robot_points[i]

        error = np.linalg.norm(predicted - expected)

        print(
            f"{point_labels[i]} | "
            f"Predicted X={predicted[0]:.3f}, Y={predicted[1]:.3f} | "
            f"Expected X={expected[0]:.3f}, Y={expected[1]:.3f} | "
            f"Error={error:.3f} mm"
        )


def pixel_to_robot(x, y, H):
    pixel = np.array([[[x, y]]], dtype=np.float32)
    robot_xy = cv2.perspectiveTransform(pixel, H)[0][0]
    return robot_xy[0], robot_xy[1]


def main():
    global clicked_points

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    window_name = "9 Point Homography Calibration"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=================================")
    print("9 POINT HOMOGRAPHY CALIBRATION")
    print("=================================")
    print("Click in this order:")
    print("P1 P2 P3")
    print("P4 P5 P6")
    print("P7 P8 P9")
    print("\nControls:")
    print("S = save calibration")
    print("R = reset")
    print("Q = quit")
    print("=================================\n")

    while True:
        frame = picam2.capture_array()

        # Picamera2 gives RGB, OpenCV uses BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display = draw_points(frame_bgr)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break

        elif key == ord("r"):
            clicked_points = []
            print("[INFO] Reset. Click again starting at P1.")

        elif key == ord("s"):
            save_homography()

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
