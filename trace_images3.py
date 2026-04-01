import cv2
import numpy as np
import sys
import os

MIN_COMPONENT_AREA = 80
KERNEL_SIZE = 5

def get_image_path():
    if len(sys.argv) < 2:
        print("Usage: python trace_images2.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    return image_path

def remove_small_components(binary, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def main():
    image_path = get_image_path()
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    print("Select the tattoo area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select Tattoo", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No ROI selected.")
        sys.exit(1)

    cropped = image[y:y+h, x:x+w].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce skin texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Blackhat helps isolate dark lines on brighter skin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    # Normalize to improve contrast
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold the dark-line response
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, clean_kernel, iterations=2)

    # Remove tiny specks
    cleaned = remove_small_components(cleaned, MIN_COMPONENT_AREA)

    # Find contours of the ink lines
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw trace on white background
    trace_white = np.full((cropped.shape[0], cropped.shape[1], 3), 255, dtype=np.uint8)

    kept = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 20:
            cv2.drawContours(trace_white, [cnt], -1, (0, 0, 0), 2)
            kept += 1

    # Overlay version too
    overlay = cropped.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 20:
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

    # Save outputs
    cv2.imwrite("01_cropped.png", cropped)
    cv2.imwrite("02_gray.png", gray)
    cv2.imwrite("03_blackhat.png", blackhat)
    cv2.imwrite("04_binary.png", binary)
    cv2.imwrite("05_cleaned.png", cleaned)
    cv2.imwrite("06_trace_white.png", trace_white)
    cv2.imwrite("07_overlay.png", overlay)

    print(f"Kept {kept} contours")
    print("Saved:")
    print(" - 01_cropped.png")
    print(" - 02_gray.png")
    print(" - 03_blackhat.png")
    print(" - 04_binary.png")
    print(" - 05_cleaned.png")
    print(" - 06_trace_white.png")
    print(" - 07_overlay.png")

    cv2.imshow("Cropped", cropped)
    cv2.imshow("Blackhat", blackhat)
    cv2.imshow("Binary", binary)
    cv2.imshow("Cleaned", cleaned)
    cv2.imshow("Trace on White", trace_white)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()