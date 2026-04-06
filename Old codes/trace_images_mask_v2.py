import cv2
import numpy as np
import sys
from pathlib import Path

def remove_small_components(binary, min_area=12):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def extract_line_mask_v2(image_path, output_prefix="mask_v2", use_roi=True):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if use_roi:
        print("Select the design area, then press ENTER or SPACE.")
        roi = cv2.selectROI("Select Design Area", img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w > 0 and h > 0:
            img = img[y:y+h, x:x+w]
        else:
            print("No ROI selected. Using full image.")

    cropped = img.copy()

    # 1) grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 2) contrast enhancement for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3) light blur to reduce skin/paper texture without killing thin lines
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4) adaptive threshold so different image areas can threshold differently
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,   # neighborhood size, must be odd
        7     # subtract constant
    )

    # 5) gentle cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 6) remove tiny noise
    cleaned = remove_small_components(cleaned, min_area=12)

    # 7) optional tiny dilation to reconnect fragile thin strokes
    reconnect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_reconnected = cv2.dilate(cleaned, reconnect_kernel, iterations=1)

    # 8) final black-on-white result
    final_mask = np.full_like(cleaned_reconnected, 255)
    final_mask[cleaned_reconnected > 0] = 0

    # Save outputs
    cropped_path = f"{output_prefix}_01_cropped.png"
    gray_path = f"{output_prefix}_02_gray.png"
    enhanced_path = f"{output_prefix}_03_enhanced.png"
    binary_path = f"{output_prefix}_04_binary.png"
    cleaned_path = f"{output_prefix}_05_cleaned.png"
    reconnect_path = f"{output_prefix}_06_reconnected.png"
    final_path = f"{output_prefix}_07_final.png"

    cv2.imwrite(cropped_path, cropped)
    cv2.imwrite(gray_path, gray)
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(binary_path, binary)
    cv2.imwrite(cleaned_path, cleaned)
    cv2.imwrite(reconnect_path, cleaned_reconnected)
    cv2.imwrite(final_path, final_mask)

    print(f"Saved: {cropped_path}")
    print(f"Saved: {gray_path}")
    print(f"Saved: {enhanced_path}")
    print(f"Saved: {binary_path}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {reconnect_path}")
    print(f"Saved: {final_path}")

    cv2.imshow("Cropped", cropped)
    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Binary", binary)
    cv2.imshow("Cleaned", cleaned)
    cv2.imshow("Reconnected", cleaned_reconnected)
    cv2.imshow("Final Mask V2", final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 trace_images_mask_v2.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    extract_line_mask_v2(image_path)