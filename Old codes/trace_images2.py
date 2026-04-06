import cv2
import numpy as np
import sys
import os

# =========================
# SETTINGS
# =========================
MIN_COMPONENT_AREA = 300
BORDER_MARGIN = 10

# =========================
# GET IMAGE PATH
# =========================
def get_image_path():
    if len(sys.argv) < 2:
        print("Usage: python trace_images2.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    return image_path

# =========================
# REMOVE SMALL NOISE
# =========================
def remove_small_components(binary, min_area=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

# =========================
# REMOVE BORDER NOISE
# =========================
def remove_border_components(binary, margin=10):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        x, y, bw, bh, _ = stats[i]

        if not (
            x <= margin or
            y <= margin or
            x + bw >= w - margin or
            y + bh >= h - margin
        ):
            cleaned[labels == i] = 255

    return cleaned

# =========================
# MAIN
# =========================
def main():
    IMAGE_PATH = get_image_path()

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image: {IMAGE_PATH}")
        return

    print("👉 Select the tattoo area and press ENTER or SPACE")
    roi = cv2.selectROI("Select Tattoo", image, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("No ROI selected.")
        return

    cropped = image[y:y+h, x:x+w]

    # =========================
    # STEP 1: LAB COLOR SPACE
    # =========================
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)

    # =========================
    # STEP 2: THRESHOLD (dark ink)
    # =========================
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # =========================
    # STEP 3: CLEANUP
    # =========================
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    cleaned = remove_small_components(cleaned, MIN_COMPONENT_AREA)
    cleaned = remove_border_components(cleaned, BORDER_MARGIN)

    # =========================
    # STEP 4: EDGE DETECTION
    # =========================
    edges = cv2.Canny(blurred, 40, 120)

    edges = cv2.bitwise_and(edges, edges, mask=cleaned)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # =========================
    # STEP 5: CONTOURS
    # =========================
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    result = cropped.copy()

    final_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            final_contours.append(approx)

    cv2.drawContours(result, final_contours, -1, (0, 255, 0), 2)

    # =========================
    # SAVE OUTPUTS
    # =========================
    cv2.imwrite("cropped.png", cropped)
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("cleaned.png", cleaned)
    cv2.imwrite("edges.png", edges)
    cv2.imwrite("final_trace.png", result)

    print("\n✅ Saved outputs:")
    print(" - cropped.png")
    print(" - binary.png")
    print(" - cleaned.png")
    print(" - edges.png")
    print(" - final_trace.png")

    # =========================
    # SHOW RESULTS
    # =========================
    cv2.imshow("Final Trace", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()