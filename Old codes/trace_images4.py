import cv2
import numpy as np
import sys
from pathlib import Path

def trace_image(image_path, output_prefix="traced", use_roi=True):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = img.copy()

    # Optional ROI selection
    if use_roi:
        print("Select the tattoo area, then press ENTER or SPACE.")
        roi = cv2.selectROI("Select Tattoo Area", img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w > 0 and h > 0:
            img = img[y:y+h, x:x+w]
        else:
            print("No ROI selected. Using full image.")

    output_overlay = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce skin texture noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Clean edges a little
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter tiny noisy contours
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 20:
            epsilon = 0.002 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            filtered_contours.append(approx)

    print(f"Found {len(filtered_contours)} contours")

    # Draw on original cropped image
    cv2.drawContours(output_overlay, filtered_contours, -1, (0, 255, 0), 2)

    # Draw clean trace on white background
    white_bg = np.full_like(img, 255)
    cv2.drawContours(white_bg, filtered_contours, -1, (0, 0, 0), 2)

    # Save outputs
    overlay_path = f"{output_prefix}_overlay.png"
    trace_path = f"{output_prefix}_white.png"
    edges_path = f"{output_prefix}_edges.png"

    cv2.imwrite(overlay_path, output_overlay)
    cv2.imwrite(trace_path, white_bg)
    cv2.imwrite(edges_path, edges)

    print(f"Saved overlay to: {overlay_path}")
    print(f"Saved white trace to: {trace_path}")
    print(f"Saved edges to: {edges_path}")

    # Show results
    cv2.imshow("Edges", edges)
    cv2.imshow("Overlay Trace", output_overlay)
    cv2.imshow("White Trace", white_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trace_images2.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    trace_image(image_path)