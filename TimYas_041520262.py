import cv2
import numpy as np
import sys
from pathlib import Path


# =========================================================
# TEMPLATE FROM YOUR WORKING SQUARE JOB (TAT.JBI)
# =========================================================

APPROACH_POINT = np.array([-2822, 58859, -1118, -1041, -34109, -89], dtype=np.float64)

P1 = np.array([-2907, 56521, -4684, -1065, -32660, -27], dtype=np.float64)   # bottom-left
P2 = np.array([-1083, 56441, -4806, -1313, -32591, -753], dtype=np.float64)  # bottom-right
P3 = np.array([-1053, 58722, -1325, -1275, -34006, -799], dtype=np.float64)  # top-right
P4 = np.array([-2854, 58833, -1199, -1038, -34058, -75], dtype=np.float64)   # top-left

DEFAULT_POINTS_PER_CONTOUR = 18
MIN_CONTOUR_AREA = 30

JOB_NAME = "JUMPTRC"
JOB_FILE = "JUMPTRC.JBI"

MOVEJ_SPEED = 0.78   # jump / reposition
MOVL_SPEED = 20.0    # draw move


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
# MULTI-CONTOUR SAMPLING
# =========================================================

def resample_closed_contour(contour_xy, num_points):
    contour = contour_xy.astype(np.float32)

    if len(contour) < 2:
        return contour

    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        return contour[:1]

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

    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])

    return sampled


def find_and_sample_all_contours(mask, points_per_contour=18, min_area=30):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contour found.")

    kept = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            kept.append(contour[:, 0, :].astype(np.float32))

    if not kept:
        raise RuntimeError("Contours found, but all were too small after filtering.")

    # sort left-to-right for more predictable jump order
    def contour_key(c):
        m = cv2.moments(c.reshape(-1, 1, 2))
        if abs(m["m00"]) < 1e-9:
            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))
        else:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        return (cx, cy)

    kept.sort(key=contour_key)

    sampled_contours = []
    for c in kept:
        sampled = resample_closed_contour(c, points_per_contour)
        sampled_contours.append(sampled)

    return sampled_contours


def save_labeled_point_image(mask, sampled_contours, output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for ci, contour in enumerate(sampled_contours):
        for pi, p in enumerate(contour):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(debug, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                debug,
                f"{ci}:{pi}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255),
                1,
            )

    cv2.imwrite(output_path, debug)
    print(f"Saved labeled point image: {output_path}")


# =========================================================
# MAP TO YOUR WORKING TAT SQUARE
# =========================================================

def bilinear_interp(u, v, p1, p2, p3, p4):
    return (
        (1 - u) * (1 - v) * p1 +
        u * (1 - v) * p2 +
        u * v * p3 +
        (1 - u) * v * p4
    )


def all_contours_bounds(sampled_contours):
    pts = np.vstack(sampled_contours)
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    return min_x, max_x, min_y, max_y


def contours_to_template_pulses(sampled_contours):
    min_x, max_x, min_y, max_y = all_contours_bounds(sampled_contours)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_contours = []
    mapping = []

    global_index = 0
    for ci, contour in enumerate(sampled_contours):
        mapped_one = []
        for pi, (x, y) in enumerate(contour):
            u = (x - min_x) / width
            v = 1.0 - ((y - min_y) / height)

            pulse_vec = bilinear_interp(u, v, P1, P2, P3, P4)
            pulse = [int(round(val)) for val in pulse_vec.tolist()]
            mapped_one.append(pulse)

            mapping.append({
                "global_index": global_index,
                "contour_index": ci,
                "point_index": pi,
                "image_x": float(x),
                "image_y": float(y),
                "u": float(u),
                "v": float(v),
                "pulse": pulse,
            })
            global_index += 1

        mapped_contours.append(mapped_one)

    return mapped_contours, mapping


def save_point_mapping(mapping, output_path="point_mapping_jumps.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping (Image -> TAT template pulse area)\n")
        f.write("=" * 78 + "\n\n")

        for item in mapping:
            f.write(f"Global Point {item['global_index']}\n")
            f.write(f"  Contour/Point : ({item['contour_index']}, {item['point_index']})\n")
            f.write(f"  Image Point   : ({item['image_x']:.2f}, {item['image_y']:.2f})\n")
            f.write(f"  Template UV   : ({item['u']:.4f}, {item['v']:.4f})\n")
            f.write(
                f"  Pulse Point   : "
                f"({item['pulse'][0]}, {item['pulse'][1]}, {item['pulse'][2]}, "
                f"{item['pulse'][3]}, {item['pulse'][4]}, {item['pulse'][5]})\n\n"
            )

    print(f"Saved point mapping: {output_path}")


# =========================================================
# JBI WRITER WITH JUMPS
# =========================================================

def write_jbi_with_jumps(mapped_contours, filename=JOB_FILE, job_name=JOB_NAME):
    all_points = [APPROACH_POINT.astype(int).tolist()]
    contour_start_indices = []
    contour_lengths = []

    for contour in mapped_contours:
        contour_start_indices.append(len(all_points))
        contour_lengths.append(len(contour))
        all_points.extend(contour)

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

    # safe move to approach first
    lines.append(f"MOVJ C00000 VJ={MOVEJ_SPEED:.2f}")

    for start_idx, contour_len in zip(contour_start_indices, contour_lengths):
        # jump to start of this disconnected piece
        lines.append(f"MOVJ C{start_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        # draw through this piece
        for point_idx in range(start_idx + 1, start_idx + contour_len):
            lines.append(f"MOVL C{point_idx:05d} V={MOVL_SPEED:.1f}")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python TimYas_jumps.py <image_path> [points_per_contour]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    points_per_contour = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_POINTS_PER_CONTOUR

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)

    sampled_contours = find_and_sample_all_contours(
        chosen_mask,
        points_per_contour=points_per_contour,
        min_area=MIN_CONTOUR_AREA,
    )

    save_labeled_point_image(chosen_mask, sampled_contours)

    mapped_contours, mapping = contours_to_template_pulses(sampled_contours)

    save_point_mapping(mapping)
    write_jbi_with_jumps(mapped_contours, filename=JOB_FILE, job_name=JOB_NAME)

    print("\nDone.")
    print(f"Found {len(sampled_contours)} disconnected contour(s).")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - point_mapping_jumps.txt")
    print(f"  - {JOB_FILE}")
    print("\nBehavior:")
    print("  - MOVJ jump to each separate contour")
    print("  - MOVL draw inside each contour")


if __name__ == "__main__":
    main()