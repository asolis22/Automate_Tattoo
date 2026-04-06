import cv2
import numpy as np
import sys
from pathlib import Path

def trace_image(image_path, output_path="traced_output.png"):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Optional: remove tiny noisy contours
    contours = [c for c in contours if cv2.contourArea(c) > 20]

    print(f"Found {len(contours)} contours")

    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(output_path, output)
    print(f"Saved traced image to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trace_image.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    trace_image(image_path)