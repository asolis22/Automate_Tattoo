import cv2
import numpy as np
import sys
from pathlib import Path


# -----------------------------
# IMAGE CLEANUP / TRACE SECTION
# -----------------------------

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def method_v1(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = remove_small_components(cleaned, min_area=20)

    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0

    return gray, binary, cleaned, final_mask


def method_v1b(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    background = cv2.GaussianBlur(gray_blur, (31, 31), 0)
    ink_response = cv2.subtract(background, gray_blur)
    ink_response = cv2.normalize(ink_response, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(ink_response, 40, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = remove_small_components(cleaned, min_area=10)

    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0

    return gray, ink_response, cleaned, final_mask


def stack_for_display(img1, img2, label1="Method 1", label2="Method 2"):
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h = max(img1.shape[0], img2.shape[0])

    def pad_to_height(img, target_h):
        if img.shape[0] == target_h:
            return img
        pad = target_h - img.shape[0]
        return cv2.copyMakeBorder(
            img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    img1 = pad_to_height(img1, h)
    img2 = pad_to_height(img2, h)

    cv2.putText(img1, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img2, label2, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return np.hstack((img1, img2))


def choose_trace_mask(img):
    print("Select the design area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select Design Area", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w > 0 and h > 0:
        cropped = img[y:y+h, x:x+w]
    else:
        print("No ROI selected. Using full image.")
        cropped = img.copy()

    gray1, binary1, cleaned1, final1 = method_v1(cropped)
    gray2, response2, cleaned2, final2 = method_v1b(cropped)

    cv2.imwrite("hybrid_cropped.png", cropped)
    cv2.imwrite("hybrid_v1_binary.png", binary1)
    cv2.imwrite("hybrid_v1_final.png", final1)
    cv2.imwrite("hybrid_v1b_response.png", response2)
    cv2.imwrite("hybrid_v1b_final.png", final2)

    compare = stack_for_display(final1, final2, "1: Thick/Fill Mode", "2: Thin/Sketch Mode")
    cv2.imshow("Choose Best Result: Press 1 or 2", compare)

    print("Press 1 for thick-line mode, or 2 for thin-sketch mode.")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord("1"):
        chosen = final1
        chosen_name = "hybrid_chosen_v1.png"
        print("Chose Method 1")
    elif key == ord("2"):
        chosen = final2
        chosen_name = "hybrid_chosen_v1b.png"
        print("Chose Method 2")
    else:
        raise RuntimeError("No valid choice made. Press 1 or 2.")

    cv2.imwrite(chosen_name, chosen)
    print(f"Saved chosen output to: {chosen_name}")
    return chosen


# -----------------------------
# SHAPE DETECTION SECTION
# -----------------------------

def detect_simple_shape(mask):
    """
    Detects the main contour and classifies it as:
    triangle, square, rectangle, circle, or unknown.
    """
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contour found in selected image.")

    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    area = cv2.contourArea(contour)
    if area < 50:
        raise RuntimeError("Detected contour is too small.")

    shape = "unknown"

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h != 0 else 0
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "square"
        else:
            shape = "rectangle"
    else:
        circularity = 4 * np.pi * area / (peri * peri) if peri != 0 else 0
        if circularity > 0.75:
            shape = "circle"

    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug, [approx], -1, (0, 0, 255), 2)
    cv2.imwrite("detected_shape_debug.png", debug)

    print(f"Detected shape: {shape}")
    print(f"Approx vertices: {len(approx)}")

    return shape


# -----------------------------
# YASKAWA JBI GENERATION
# -----------------------------

def build_pulse_template(shape):
    """
    Returns safe in-air pulse points.
    These are TEMPLATE points only.
    Tune these to your robot/cell after first successful test.
    """

    # Base point from your exported examples style
    # Keep these conservative and in the air.
    base = [-3626, 25307, 0, 0, -64000, -407]

    def pt(j1, j2, j3=0, j4=0, j5=-64000, j6=-407):
        return [j1, j2, j3, j4, j5, j6]

    if shape == "triangle":
        return [
            pt(base[0], base[1]),
            pt(base[0] + 1200, base[1] + 1800),
            pt(base[0] + 2400, base[1]),
            pt(base[0], base[1]),
        ]

    if shape == "square":
        return [
            pt(base[0], base[1]),
            pt(base[0] + 1600, base[1]),
            pt(base[0] + 1600, base[1] + 1600),
            pt(base[0], base[1] + 1600),
            pt(base[0], base[1]),
        ]

    if shape == "rectangle":
        return [
            pt(base[0], base[1]),
            pt(base[0] + 2200, base[1]),
            pt(base[0] + 2200, base[1] + 1200),
            pt(base[0], base[1] + 1200),
            pt(base[0], base[1]),
        ]

    if shape == "circle":
        # Octagon approximation for now
        return [
            pt(base[0] + 800, base[1]),
            pt(base[0] + 1400, base[1] + 600),
            pt(base[0] + 1400, base[1] + 1400),
            pt(base[0] + 800, base[1] + 2000),
            pt(base[0], base[1] + 1400),
            pt(base[0] - 600, base[1] + 800),
            pt(base[0], base[1]),
            pt(base[0] + 800, base[1]),
        ]

    # fallback
    return [
        pt(base[0], base[1]),
        pt(base[0] + 1600, base[1]),
        pt(base[0] + 1600, base[1] + 1600),
        pt(base[0], base[1] + 1600),
        pt(base[0], base[1]),
    ]


def write_jbi_pulse_job(points, filename="TRACE.JBI", job_name="TRACE1"):
    """
    Writes a simple Yaskawa PULSE job based on the exported style
    you've shown from your controller.
    """
    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(points)},0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    for i, p in enumerate(points):
        lines.append(f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")

    lines.append("//INST")
    lines.append("///DATE 2026/04/01 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")

    # start safely at first point
    lines.append("MOVJ C00000 VJ=5.00")

    for i in range(1, len(points)):
        lines.append(f"MOVL C{i:05d} V=100.0")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# -----------------------------
# MAIN
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tim_to_yaskawa.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)
    shape = detect_simple_shape(chosen_mask)

    points = build_pulse_template(shape)
    write_jbi_pulse_job(points, filename="TRACE.JBI", job_name="TRACE1")

    print("\nDone.")
    print("1. Copy TRACE.JBI to your USB.")
    print("2. Load it on the Yaskawa.")
    print("3. Test at very low speed in the air first.")


if __name__ == "__main__":
    main()