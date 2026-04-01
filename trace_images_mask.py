import cv2
import numpy as np
import sys
from pathlib import Path

def remove_small_components(binary, min_area=30):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def extract_line_mask(image_path, output_prefix="mask", use_roi=True):
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

    # Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce skin/paper texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold dark lines directly
    # Invert so dark lines become white in mask
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove tiny specks
    cleaned = remove_small_components(cleaned, min_area=20)

    # Convert mask to black lines on white background
    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0

    # Save outputs
    cropped_path = f"{output_prefix}_cropped.png"
    gray_path = f"{output_prefix}_gray.png"
    binary_path = f"{output_prefix}_binary.png"
    cleaned_path = f"{output_prefix}_cleaned.png"
    final_path = f"{output_prefix}_final.png"

    cv2.imwrite(cropped_path, cropped)
    cv2.imwrite(gray_path, gray)
    cv2.imwrite(binary_path, binary)
    cv2.imwrite(cleaned_path, cleaned)
    cv2.imwrite(final_path, final_mask)

    print(f"Saved: {cropped_path}")
    print(f"Saved: {gray_path}")
    print(f"Saved: {binary_path}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {final_path}")

    cv2.imshow("Cropped", cropped)
    cv2.imshow("Binary", binary)
    cv2.imshow("Cleaned", cleaned)
    cv2.imshow("Final Mask", final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 trace_images_mask.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    extract_line_mask(image_path)