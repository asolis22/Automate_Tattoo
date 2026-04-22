
import cv2
import numpy as np
import sys
from pathlib import Path


# =========================================================
# TEMPLATE FROM YOUR WORKING SQUARE JOB (TAT.JBI)
# Map any design into THIS exact successful drawing area.
# Corner order follows the square:
#   P1 = bottom-left
#   P2 = bottom-right
#   P3 = top-right
#   P4 = top-left
# Approach point is the safe move-in point from the job.
# =========================================================

APPROACH_POINT = np.array([-2822, 58859, -1118, -1041, -34109, -89], dtype=np.float64)

P1 = np.array([-2907, 56521, -4684, -1065, -32660, -27], dtype=np.float64)   # C00001
P2 = np.array([-1083, 56441, -4806, -1313, -32591, -753], dtype=np.float64)  # C00002
P3 = np.array([-1053, 58722, -1325, -1275, -34006, -799], dtype=np.float64)  # C00003
P4 = np.array([-2854, 58833, -1199, -1038, -34058, -75], dtype=np.float64)   # C00004

DEFAULT_POINTS = 20
JOB_NAME = "BUTTERFLY"
JOB_FILE = "BUTTERFLY.JBI"

# Since the square job used MOVJ and worked, keep that same style first.
MOVEJ_SPEED = 0.78


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
# SAMPLE POINTS BY DISTANCE
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
# NORMALIZED IMAGE -> WORKING SQUARE TEMPLATE (pulse space)
# =========================================================

def bilinear_interp(u, v, p1, p2, p3, p4):
    """
    Map normalized square coordinates (u, v) into the exact pulse-space
    quadrilateral from the working TAT square.
    u: left->right in [0,1]
    v: bottom->top in [0,1]
    """
    return (
        (1 - u) * (1 - v) * p1 +
        u * (1 - v) * p2 +
        u * v * p3 +
        (1 - u) * v * p4
    )


def image_points_to_template_pulses(sampled_points):
    pts = sampled_points.astype(np.float32)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapping = []
    pulse_points = []

    for i, (x, y) in enumerate(pts):
        # normalize image to [0,1]
        u = (x - min_x) / width

        # flip y so image top/bottom becomes bottom/top of drawing square
        v = 1.0 - ((y - min_y) / height)

        pulse_vec = bilinear_interp(u, v, P1, P2, P3, P4)
        pulse = [int(round(val)) for val in pulse_vec.tolist()]
        pulse_points.append(pulse)

        mapping.append({
            "index": i,
            "image_x": float(x),
            "image_y": float(y),
            "u": float(u),
            "v": float(v),
            "pulse": pulse,
        })

    return pulse_points, mapping


def save_point_mapping(mapping, output_path="point_mapping_template.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping (Image -> TAT template pulse area)\n")
        f.write("=" * 72 + "\n\n")

        for item in mapping:
            f.write(f"Point {item['index']}\n")
            f.write(f"  Image Point : ({item['image_x']:.2f}, {item['image_y']:.2f})\n")
            f.write(f"  Template UV : ({item['u']:.4f}, {item['v']:.4f})\n")
            f.write(
                f"  Pulse Point : "
                f"({item['pulse'][0]}, {item['pulse'][1]}, {item['pulse'][2]}, "
                f"{item['pulse'][3]}, {item['pulse'][4]}, {item['pulse'][5]})\n\n"
            )

    print(f"Saved point mapping: {output_path}")


# =========================================================
# JBI WRITER
# =========================================================

def write_jbi_from_template(points, filename=JOB_FILE, job_name=JOB_NAME):
    all_points = [APPROACH_POINT.astype(int).tolist()] + points

    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(all_points)},0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    for i, p in enumerate(all_points):
        lines.append(f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")

    lines.append("//INST")
    lines.append("///DATE 2026/04/15 15:34")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")

    # Use the exact successful motion style/speed
    for i in range(len(all_points)):
        lines.append(f"MOVJ C{i:05d} VJ={MOVEJ_SPEED:.2f}")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python TimYas_from_TAT_template.py <image_path> [num_points]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_POINTS

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)
    sampled_points = sample_contour_points_by_distance(chosen_mask, num_points=num_points)

    save_labeled_point_image(chosen_mask, sampled_points)

    pulse_points, mapping = image_points_to_template_pulses(sampled_points)

    save_point_mapping(mapping)
    write_jbi_from_template(pulse_points, filename=JOB_FILE, job_name=JOB_NAME)

    print("\nDone.")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - point_mapping_template.txt")
    print(f"  - {JOB_FILE}")
    print("\nTemplate source:")
    print("  - approach point + square corners taken from your working TAT square job")
    print("  - all designs are mapped into that same successful drawing area")


if __name__ == "__main__":
    main()
