import cv2
import numpy as np
import sys
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================

SAFE_BASE = [-3626, 25307, 0, 0, -64000, -407]   # replace with your real safe in-air pose

# Make the drawing MUCH bigger here
PULSE_WIDTH = 12000
PULSE_HEIGHT = 12000

# More points = smoother tracing
DEFAULT_POINTS = 20

JOB_NAME = "BIGSLOW"
JOB_FILE = "BIGSLOW.JBI"

# Much slower motion
MOVEJ_SPEED = 2.00
MOVL_SPEED = 20.0


# =========================================================
# IMAGE CLEANUP
# =========================================================

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
    print("Saved chosen_mask.png")
    return chosen


# =========================================================
# CONTOUR SAMPLING BY DISTANCE
# =========================================================

def sample_contour_points_by_distance(mask, num_points=20):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contour found.")

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :].astype(np.float32)

    if len(contour) < 2:
        raise RuntimeError("Contour too small.")

    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        raise RuntimeError("Contour length is zero.")

    targets = np.linspace(0, total_length, num_points, endpoint=False)

    sampled = []
    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(idx, 0), len(seg_lengths) - 1)

        seg_start = contour[idx]
        seg_end = contour[idx + 1]
        seg_len = seg_lengths[idx]

        if seg_len == 0:
            p = seg_start
        else:
            local_t = (t - cumulative[idx]) / seg_len
            p = seg_start + local_t * (seg_end - seg_start)

        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)
    sampled = np.vstack([sampled, sampled[0]])  # close loop
    return sampled.astype(np.float32)


def save_labeled_point_image(mask, sampled_points, output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, p in enumerate(sampled_points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(debug, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(debug, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imwrite(output_path, debug)
    print(f"Saved labeled point image: {output_path}")


# =========================================================
# IMAGE -> PULSE MAPPING
# =========================================================

def image_points_to_pulses(sampled_points, safe_base, pulse_width=12000, pulse_height=12000):
    pts = sampled_points.astype(np.float32)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    j1, j2, j3, j4, j5, j6 = safe_base

    mapping = []
    pulse_points = []

    for i, (x, y) in enumerate(pts):
        nx = (x - min_x) / width
        ny = (y - min_y) / height

        dx = int(nx * pulse_width)
        dy = int(ny * pulse_height)

        # Keep j3-j6 fixed to help keep the motion in roughly the same floating plane
        pulse = [j1 + dx, j2 + dy, j3, j4, j5, j6]
        pulse_points.append(pulse)

        mapping.append({
            "index": i,
            "image_x": float(x),
            "image_y": float(y),
            "norm_x": float(nx),
            "norm_y": float(ny),
            "pulse": pulse,
        })

    return pulse_points, mapping


def save_point_mapping(mapping, output_path="point_mapping.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping\n")
        f.write("=" * 72 + "\n\n")

        for item in mapping:
            f.write(f"Point {item['index']}\n")
            f.write(f"  Image Point      : ({item['image_x']:.2f}, {item['image_y']:.2f})\n")
            f.write(f"  Normalized Point : ({item['norm_x']:.4f}, {item['norm_y']:.4f})\n")
            f.write(
                f"  Pulse Point      : "
                f"({item['pulse'][0]}, {item['pulse'][1]}, {item['pulse'][2]}, "
                f"{item['pulse'][3]}, {item['pulse'][4]}, {item['pulse'][5]})\n\n"
            )

    print(f"Saved point mapping: {output_path}")


# =========================================================
# JBI WRITER
# =========================================================

def write_jbi_pulse_job(points, filename=JOB_FILE, job_name=JOB_NAME):
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
    lines.append("///DATE 2026/04/08 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.append(f"MOVJ C00000 VJ={MOVEJ_SPEED:.2f}")

    for i in range(1, len(points)):
        lines.append(f"MOVL C{i:05d} V={MOVL_SPEED:.1f}")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python Tim_to_yaska_bigslow.py <image_path> [num_points]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_POINTS

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)
    sampled_points = sample_contour_points_by_distance(chosen_mask, num_points=num_points)

    save_labeled_point_image(chosen_mask, sampled_points)

    pulse_points, mapping = image_points_to_pulses(
        sampled_points,
        safe_base=SAFE_BASE,
        pulse_width=PULSE_WIDTH,
        pulse_height=PULSE_HEIGHT,
    )

    save_point_mapping(mapping)
    write_jbi_pulse_job(pulse_points, filename=JOB_FILE, job_name=JOB_NAME)

    print("\nDone.")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - point_mapping.txt")
    print(f"  - {JOB_FILE}")
    print("\nCurrent settings:")
    print(f"  PULSE_WIDTH  = {PULSE_WIDTH}")
    print(f"  PULSE_HEIGHT = {PULSE_HEIGHT}")
    print(f"  MOVEJ_SPEED  = {MOVEJ_SPEED}")
    print(f"  MOVL_SPEED   = {MOVL_SPEED}")


if __name__ == "__main__":
    main()