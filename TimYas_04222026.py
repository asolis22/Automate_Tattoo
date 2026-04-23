import cv2
import numpy as np
import sys
from pathlib import Path

# =========================================================
# TEMPLATE FROM YOUR WORKING TAT.JBI
# =========================================================

APPROACH_POINT = np.array([-2822, 58859, -1118, -1041, -34109, -89], dtype=np.float64)

P1 = np.array([-2907, 56521, -4684, -1065, -32660, -27], dtype=np.float64)   # bottom-left
P2 = np.array([-1083, 56441, -4806, -1313, -32591, -753], dtype=np.float64)  # bottom-right
P3 = np.array([-1053, 58722, -1325, -1275, -34006, -799], dtype=np.float64)  # top-right
P4 = np.array([-2854, 58833, -1199, -1038, -34058, -75], dtype=np.float64)   # top-left

# =========================================================
# SCALE DRAWING AREA
# =========================================================
SCALE = 3  # try 1.3 first, then 1.4, then 1.5

center = (P1 + P2 + P3 + P4) / 4.0

P1 = center + (P1 - center) * SCALE
P2 = center + (P2 - center) * SCALE
P3 = center + (P3 - center) * SCALE
P4 = center + (P4 - center) * SCALE

JOB_NAME = "CONTOUR2"
JOB_FILE = "CONTOUR2.JBI"

# =========================================================
# TUNING
# =========================================================

MOVEJ_SPEED = 0.78     # jump / reposition
MOVL_SPEED = 8.0       # slower draw for smoother lines
LIFT_J3 = 1400         # slightly bigger pen-up lift
MIN_CONTOUR_AREA = 30  # ignore tiny junk
POINTS_PER_CONTOUR = 60

# If True, all moves stay lifted for safe air-preview
AIR_ONLY_MODE = False

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
    cropped = img[y:y+h, x:x+w] if (w > 0 and h > 0) else img.copy()

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
# CONTOUR TRACING
# =========================================================

def resample_closed_contour(points, target_points=60):
    points = points.astype(np.float32)

    if len(points) < 2:
        return points

    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]

    if total <= 0:
        return points[:1]

    targets = np.linspace(0, total, target_points, endpoint=False)
    sampled = []

    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(idx, 0), len(seg_lengths) - 1)

        start = points[idx]
        end = points[idx + 1]
        seg_len = seg_lengths[idx]

        if seg_len == 0:
            p = start
        else:
            alpha = (t - cumulative[idx]) / seg_len
            p = start + alpha * (end - start)

        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)

    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])

    return sampled


def extract_contours(mask, min_area=30, points_per_contour=60):
    inv = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contours found.")

    kept = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            pts = contour[:, 0, :].astype(np.float32)
            pts = resample_closed_contour(pts, target_points=points_per_contour)
            if len(pts) >= 3:
                kept.append(pts)

    if not kept:
        raise RuntimeError("Contours were found, but all were too small after filtering.")

    return kept


# =========================================================
# MAP TO TEMPLATE
# =========================================================

def bilinear_interp(u, v, p1, p2, p3, p4):
    return (
        (1 - u) * (1 - v) * p1 +
        u * (1 - v) * p2 +
        u * v * p3 +
        (1 - u) * v * p4
    )


def all_contours_bounds(contours):
    pts = np.vstack(contours)
    return np.min(pts[:, 0]), np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 1])


def contours_to_template_pulses(contours):
    min_x, max_x, min_y, max_y = all_contours_bounds(contours)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_contours = []
    mapping = []
    gi = 0

    for ci, contour in enumerate(contours):
        mapped = []
        for pi, (x, y) in enumerate(contour):
            u = (x - min_x) / width
            v = 1.0 - ((y - min_y) / height)

            pulse_vec = bilinear_interp(u, v, P1, P2, P3, P4)
            pulse = [int(round(vv)) for vv in pulse_vec.tolist()]
            mapped.append(pulse)

            mapping.append({
                "global_index": gi,
                "contour_index": ci,
                "point_index": pi,
                "image_x": float(x),
                "image_y": float(y),
                "u": float(u),
                "v": float(v),
                "pulse": pulse,
            })
            gi += 1

        mapped_contours.append(mapped)

    return mapped_contours, mapping


# =========================================================
# BETTER CONTOUR ORDERING
# =========================================================

def contour_centroid(contour):
    pts = np.array(contour, dtype=np.float32)
    return np.array([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))], dtype=np.float32)


def rotate_contour_to_best_start(contour, ref_point):
    """
    Rotate a closed contour so it starts at the point closest to ref_point.
    """
    if len(contour) <= 1:
        return contour

    pts = np.array(contour, dtype=np.int32)
    # avoid duplicated last point for search/rotation
    closed = np.array_equal(pts[0], pts[-1])
    core = pts[:-1] if closed else pts

    ref = np.array(ref_point, dtype=np.float32)
    d2 = np.sum((core.astype(np.float32) - ref) ** 2, axis=1)
    best_idx = int(np.argmin(d2))

    rotated = np.vstack([core[best_idx:], core[:best_idx]])
    rotated = np.vstack([rotated, rotated[0]])
    return rotated.tolist()


def reorder_contours_nearest(mapped_contours):
    """
    Choose next contour by nearest start-to-current distance.
    Also rotate each contour to start at the best point.
    """
    remaining = [list(map(list, contour)) for contour in mapped_contours]
    if not remaining:
        return []

    ordered = []
    current_ref = APPROACH_POINT.astype(np.int32).tolist()

    while remaining:
        best_i = None
        best_rotated = None
        best_dist = None

        for i, contour in enumerate(remaining):
            rotated = rotate_contour_to_best_start(contour, current_ref[:2] + [0,0,0,0] if False else current_ref)
            start = np.array(rotated[0], dtype=np.float32)
            ref = np.array(current_ref, dtype=np.float32)
            dist = float(np.sum((start - ref) ** 2))

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_i = i
                best_rotated = rotated

        chosen = best_rotated
        ordered.append(chosen)
        current_ref = chosen[-1]
        remaining.pop(best_i)

    return ordered


# =========================================================
# DEBUG OUTPUTS
# =========================================================

def save_labeled_contour_image(mask, contours, output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for ci, contour in enumerate(contours):
        for pi, p in enumerate(contour):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                debug,
                f"{ci}:{pi}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (0, 0, 255),
                1,
            )

    cv2.imwrite(output_path, debug)
    print(f"Saved labeled contour image: {output_path}")


def save_travel_order_image(mask, contours, output_path="travel_order.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for ci, contour in enumerate(contours):
        color = (0, 255, 0)
        pts = np.array(contour, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(debug, tuple(pts[i][:2]), tuple(pts[i + 1][:2]), color, 1)
        start = tuple(pts[0][:2])
        cv2.circle(debug, start, 7, (255, 0, 0), -1)
        cv2.putText(debug, str(ci), (start[0] + 4, start[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(output_path, debug)
    print(f"Saved travel order image: {output_path}")


def save_point_mapping(mapping, output_path="point_mapping_contours.txt"):
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
# LIFT / JUMPS
# =========================================================

def make_lifted_pulse_point(draw_pulse):
    lifted = list(draw_pulse)
    lifted[2] += LIFT_J3
    return lifted


# =========================================================
# JBI WRITER
# =========================================================

def write_jbi_contour_trace(mapped_contours, filename=JOB_FILE, job_name=JOB_NAME):
    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append(p)
        return idx

    approach = APPROACH_POINT.astype(int).tolist()
    if AIR_ONLY_MODE:
        approach = make_lifted_pulse_point(approach)

    approach_idx = add_point(approach)
    instructions.append(f"MOVJ C{approach_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    for contour in mapped_contours:
        if len(contour) < 2:
            continue

        start = contour[0]
        end = contour[-1]

        lifted_start = make_lifted_pulse_point(start)
        lifted_end = make_lifted_pulse_point(end)

        if AIR_ONLY_MODE:
            contour = [make_lifted_pulse_point(p) for p in contour]

        lifted_start_idx = add_point(lifted_start)
        start_idx = add_point(contour[0])

        instructions.append(f"MOVJ C{lifted_start_idx:05d} VJ={MOVEJ_SPEED:.2f}")
        instructions.append(f"MOVJ C{start_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        for p in contour[1:]:
            idx = add_point(p)
            instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

        lifted_end_idx = add_point(lifted_end)
        instructions.append(f"MOVJ C{lifted_end_idx:05d} VJ={MOVEJ_SPEED:.2f}")

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
    lines.append("///DATE 2026/04/22 17:20")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python TimYas_contour_trace.py <image_path> [points_per_contour]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    points_per_contour = int(sys.argv[2]) if len(sys.argv) >= 3 else POINTS_PER_CONTOUR

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)

    contours = extract_contours(
        chosen_mask,
        min_area=MIN_CONTOUR_AREA,
        points_per_contour=points_per_contour
    )

    save_labeled_contour_image(chosen_mask, contours)

    mapped_contours, mapping = contours_to_template_pulses(contours)
    ordered_contours = reorder_contours_nearest(mapped_contours)

    save_travel_order_image(chosen_mask, ordered_contours)
    save_point_mapping(mapping)
    write_jbi_contour_trace(ordered_contours, filename=JOB_FILE, job_name=JOB_NAME)

    print("\nDone.")
    print(f"Found {len(contours)} contour(s).")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - travel_order.png")
    print("  - point_mapping_contours.txt")
    print(f"  - {JOB_FILE}")
    print("\nBehavior:")
    print("  - follows outside visible outlines")
    print("  - MOVL draws each contour in order")
    print("  - MOVJ lifts between separate contours")
    print("  - contours are reordered by nearest-next travel")
    print(f"  - safe guess lift: J3 += {LIFT_J3}")
    print(f"  - points per contour: {points_per_contour}")
    print(f"  - AIR_ONLY_MODE: {AIR_ONLY_MODE}")


if __name__ == "__main__":
    main()
