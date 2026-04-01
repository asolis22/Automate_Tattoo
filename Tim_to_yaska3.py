import cv2
import numpy as np
import sys
from pathlib import Path


# -----------------------------
# IMAGE CLEANUP
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
    return final_mask


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
    return final_mask


def stack_for_display(img1, img2):
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

    cv2.putText(img1, "1: Thick/Fill Mode", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img2, "2: Thin/Sketch Mode", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

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

    final1 = method_v1(cropped)
    final2 = method_v1b(cropped)

    compare = stack_for_display(final1, final2)
    cv2.imshow("Choose Best Result: Press 1 or 2", compare)

    print("Press 1 for thick-line mode, or 2 for thin-sketch mode.")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord("1"):
        chosen = final1
        print("Chose Method 1")
    elif key == ord("2"):
        chosen = final2
        print("Chose Method 2")
    else:
        raise RuntimeError("No valid choice made. Press 1 or 2.")

    cv2.imwrite("chosen_mask.png", chosen)
    return chosen


# -----------------------------
# CONTOUR SAMPLING
# -----------------------------

def sample_contour_points(mask, num_points=12):
    """
    Finds the main contour and samples a small number of evenly spaced points.
    """
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contour found.")

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]  # shape: (N, 2)

    if len(contour) < num_points:
        num_points = len(contour)

    idx = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    sampled = contour[idx]

    # close the loop
    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])

    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for p in sampled:
        cv2.circle(debug, tuple(int(v) for v in p), 8, (0, 255, 0), -1)

    cv2.imwrite("sampled_points_debug.png", debug)
    return sampled


# -----------------------------
# IMAGE POINTS -> PULSE POINTS
# -----------------------------

def image_points_to_pulses(sampled_points, safe_base, pulse_width=3000, pulse_height=3000):
    """
    Maps sampled image points into a relative pulse box around a known safe base point.
    """
    pts = sampled_points.astype(np.float32)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    normalized = []
    for x, y in pts:
        nx = (x - min_x) / width
        ny = (y - min_y) / height
        normalized.append((nx, ny))

    j1, j2, j3, j4, j5, j6 = safe_base

    pulse_points = []
    for nx, ny in normalized:
        dx = int(nx * pulse_width)
        dy = int(ny * pulse_height)

        pulse_points.append([
            j1 + dx,
            j2 + dy,
            j3,
            j4,
            j5,
            j6,
        ])

    return pulse_points


# -----------------------------
# JBI WRITER
# -----------------------------

def write_jbi_pulse_job(points, filename="TRACEPTS.JBI", job_name="TRACEPTS"):
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
    lines.append("MOVJ C00000 VJ=5.00")

    for i in range(1, len(points)):
        lines.append(f"MOVL C{i:05d} V=80.0")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# -----------------------------
# MAIN
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python TIM_to_sampled_YASKAWA.py <image_path> [num_points]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else 12

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)
    sampled_points = sample_contour_points(chosen_mask, num_points=num_points)

    # IMPORTANT:
    # Replace this with your real safe floating pulse pose if needed.
    safe_base = [-3626, 25307, 0, 0, -64000, -407]

    pulse_points = image_points_to_pulses(
        sampled_points,
        safe_base=safe_base,
        pulse_width=3000,
        pulse_height=3000,
    )

    write_jbi_pulse_job(pulse_points, filename="TRACEPTS.JBI", job_name="TRACEPTS")

    print("\nDone.")
    print(f"Generated {len(pulse_points)} robot points.")
    print("Saved sampled_points_debug.png so you can see the green-dot style point selection.")
    print("Test TRACEPTS.JBI in the air at very low speed first.")


if __name__ == "__main__":
    main()