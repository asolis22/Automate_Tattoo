import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================

# Minimum contour area to reject tiny noise
MIN_AREA = 8000

# HSV thresholds for peach/fake skin on blue background
# These may need small tuning depending on lighting.
LOWER_SKIN = np.array([5, 20, 120])
UPPER_SKIN = np.array([35, 170, 255])

# Morphology kernel size
KERNEL_SIZE = 5


# =========================================================
# DETECTION FUNCTION
# =========================================================
def detect_fake_skin_rotated_box(frame):
    """
    Detect fake skin sheet and return:
    - annotated image
    - mask
    - detection info dict or None
    """

    output = frame.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold for fake skin color
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    # Clean noise
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Optional blur to smooth edges a bit
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return output, mask, None

    # Keep largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_AREA:
        return output, mask, None

    # Rotated bounding box
    rect = cv2.minAreaRect(largest)
    # rect = ((cx, cy), (w, h), angle)

    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Draw rotated rectangle
    cv2.drawContours(output, [box], 0, (0, 255, 0), 3)

    # Draw contour too, if you want visual confirmation
    cv2.drawContours(output, [largest], -1, (255, 0, 0), 2)

    # Draw center
    center_pt = (int(cx), int(cy))
    cv2.circle(output, center_pt, 7, (0, 0, 255), -1)

    # Label corners
    for i, pt in enumerate(box):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(output, (px, py), 6, (0, 255, 255), -1)
        cv2.putText(
            output,
            f"P{i+1}",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    # Put text info on image
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
        f"W: {int(w)}  H: {int(h)}  Angle: {angle:.2f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    detection_info = {
        "center": (float(cx), float(cy)),
        "width": float(w),
        "height": float(h),
        "angle": float(angle),
        "box_points": box.tolist(),
        "area": float(area),
    }

    return output, mask, detection_info


# =========================================================
# CAMERA MODE
# =========================================================
def run_camera_mode():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        print("Make sure Terminal / VS Code has Camera permission.")
        return

    print("\nCamera mode started.")
    print("Press 'q' to quit.")
    print("Press 's' to save current detection image.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame from camera.")
            break

        annotated, mask, info = detect_fake_skin_rotated_box(frame)

        if info is not None:
            print("\rDetected | Center: {} | W: {:.1f} | H: {:.1f} | Angle: {:.2f}      ".format(
                tuple(map(int, info["center"])),
                info["width"],
                info["height"],
                info["angle"]
            ), end="")
        else:
            print("\rNo fake skin detected.                                   ", end="")

        cv2.imshow("Detection - Rotated Bounding Box", annotated)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("detected_fake_skin_result.jpg", annotated)
            cv2.imwrite("detected_fake_skin_mask.jpg", mask)
            print("\nSaved detected_fake_skin_result.jpg and detected_fake_skin_mask.jpg")

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# IMAGE MODE
# =========================================================
def run_image_mode():
    image_path = input("Enter image filename or full path: ").strip()

    if not os.path.exists(image_path):
        print("ERROR: File not found.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("ERROR: Could not load image.")
        return

    annotated, mask, info = detect_fake_skin_rotated_box(frame)

    if info is None:
        print("No fake skin detected.")
    else:
        print("\nDetected fake skin.")
        print(f"Center: ({int(info['center'][0])}, {int(info['center'][1])})")
        print(f"Width: {info['width']:.2f}")
        print(f"Height: {info['height']:.2f}")
        print(f"Angle: {info['angle']:.2f}")
        print("Rotated box points:")
        for i, pt in enumerate(info["box_points"], start=1):
            print(f"  P{i}: {pt}")

    cv2.imwrite("detected_fake_skin_result.jpg", annotated)
    cv2.imwrite("detected_fake_skin_mask.jpg", mask)

    print("\nSaved:")
    print(" - detected_fake_skin_result.jpg")
    print(" - detected_fake_skin_mask.jpg")

    cv2.imshow("Detection - Rotated Bounding Box", annotated)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================
def main():
    print("Choose input mode:")
    print("1 = Live camera")
    print("2 = Upload image file")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_camera_mode()
    elif choice == "2":
        run_image_mode()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()