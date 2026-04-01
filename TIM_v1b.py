import cv2
import numpy as np
import sys
from pathlib import Path

def remove_small_components(binary, min_area=10):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def extract_line_mask(image_path, output_prefix="mask_v1b", use_roi=True):
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

    # 2) small blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3) estimate smooth background with a much larger blur
    background = cv2.GaussianBlur(gray_blur, (31, 31), 0)

    # 4) subtract image from background so dark lines become bright
    ink_response = cv2.subtract(background, gray_blur)

    # 5) normalize for stronger contrast
    ink_response = cv2.normalize(ink_response, None, 0, 255, cv2.NORM_MINMAX)

    # 6) threshold dark-line response
    _, binary = cv2.threshold(ink_response, 40, 255, cv2.THRESH_BINARY)

    # 7) gentle cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8) remove tiny noise
    cleaned = remove_small_components(cleaned, min_area=10)

    # 9) final black-on-white result
    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0

    # Save outputs
    cv2.imwrite(f"{output_prefix}_01_cropped.png", cropped)
    cv2.imwrite(f"{output_prefix}_02_gray.png", gray)
    cv2.imwrite(f"{output_prefix}_03_background.png", background)
    cv2.imwrite(f"{output_prefix}_04_ink_response.png", ink_response)
    cv2.imwrite(f"{output_prefix}_05_binary.png", binary)
    cv2.imwrite(f"{output_prefix}_06_cleaned.png", cleaned)
    cv2.imwrite(f"{output_prefix}_07_final.png", final_mask)

    print("Saved outputs:")
    print(f" - {output_prefix}_01_cropped.png")
    print(f" - {output_prefix}_02_gray.png")
    print(f" - {output_prefix}_03_background.png")
    print(f" - {output_prefix}_04_ink_response.png")
    print(f" - {output_prefix}_05_binary.png")
    print(f" - {output_prefix}_06_cleaned.png")
    print(f" - {output_prefix}_07_final.png")

    cv2.imshow("Cropped", cropped)
    cv2.imshow("Ink Response", ink_response)
    cv2.imshow("Binary", binary)
    cv2.imshow("Cleaned", cleaned)
    cv2.imshow("Final Mask", final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 trace_images_mask_v1b.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    extract_line_mask(image_path)