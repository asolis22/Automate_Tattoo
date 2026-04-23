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

# Optional scaling of whole drawing area
SCALE = 4.0
center = (P1 + P2 + P3 + P4) / 4.0
P1 = center + (P1 - center) * SCALE
P2 = center + (P2 - center) * SCALE
P3 = center + (P3 - center) * SCALE
P4 = center + (P4 - center) * SCALE

JOB_NAME = "BLACKRUN"
JOB_FILE = "BLACKRUN.JBI"

# =========================================================
# TUNING
# =========================================================

MOVEJ_SPEED = 0.78
MOVL_SPEED = 8.0
LIFT_J3 = 1400

# Image cleanup / fill behavior
MIN_COMPONENT_AREA = 8
ROW_STEP = 8          # smaller = more fill lines, bigger = faster/less dense
MIN_RUN_LENGTH = 6    # ignore tiny black specks
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
# BLACK-RUN EXTRACTION
# =========================================================

def extract_black_runs(mask, row_step=3, min_run_length=2):
    """
    Scan each row and collect continuous black runs.
    Output format:
      [
        [(x0,y), (x1,y)],
        [(x0,y), (x1,y)],
        ...
      ]
    """
    h, w = mask.shape
    runs = []

    for y in range(0, h, row_step):
        row = mask[y, :]
        black = (row == 0).astype(np.uint8)

        x = 0
        while x < w:
            if black[x] == 1:
                start = x
                while x < w and black[x] == 1:
                    x += 1
                end = x - 1

                if (end - start + 1) >= min_run_length:
                    runs.append([(start, y), (end, y)])
            else:
                x += 1

    return runs


def save_run_debug_image(mask, runs, output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, run in enumerate(runs):
        (x0, y0), (x1, y1) = run
        cv2.line(debug, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(debug, (x0, y0), 3, (255, 0, 0), -1)
        cv2.circle(debug, (x1, y1), 3, (0, 0, 255), -1)

        if i < 200:
            cv2.putText(debug, str(i), (x0 + 2, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

    cv2.imwrite(output_path, debug)
    print(f"Saved run debug image: {output_path}")

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


def mask_bounds_from_runs(runs):
    pts = []
    for run in runs:
        pts.extend(run)
    pts = np.array(pts, dtype=np.float32)
    return np.min(pts[:, 0]), np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 1])


def runs_to_template_pulses(runs):
    min_x, max_x, min_y, max_y = mask_bounds_from_runs(runs)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_runs = []
    mapping = []
    gi = 0

    for ri, run in enumerate(runs):
        mapped = []
        for pi, (x, y) in enumerate(run):
            u = (x - min_x) / width
            v = 1.0 - ((y - min_y) / height)

            pulse_vec = bilinear_interp(u, v, P1, P2, P3, P4)
            pulse = [int(round(vv)) for vv in pulse_vec.tolist()]
            mapped.append(pulse)

            mapping.append({
                "global_index": gi,
                "run_index": ri,
                "point_index": pi,
                "image_x": float(x),
                "image_y": float(y),
                "u": float(u),
                "v": float(v),
                "pulse": pulse,
            })
            gi += 1

        mapped_runs.append(mapped)

    return mapped_runs, mapping


def save_point_mapping(mapping, output_path="point_mapping_runs.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping (Black runs -> TAT template pulse area)\n")
        f.write("=" * 78 + "\n\n")

        for item in mapping:
            f.write(f"Global Point {item['global_index']}\n")
            f.write(f"  Run/Point     : ({item['run_index']}, {item['point_index']})\n")
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

def write_jbi_black_runs(mapped_runs, filename=JOB_FILE, job_name=JOB_NAME):
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

    for run in mapped_runs:
        if len(run) < 2:
            continue

        start = run[0]
        end = run[-1]

        lifted_start = make_lifted_pulse_point(start)
        lifted_end = make_lifted_pulse_point(end)

        if AIR_ONLY_MODE:
            run = [make_lifted_pulse_point(p) for p in run]

        lifted_start_idx = add_point(lifted_start)
        start_idx = add_point(run[0])

        # pen up travel
        instructions.append(f"MOVJ C{lifted_start_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        # pen down
        instructions.append(f"MOVJ C{start_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        # draw black run
        for p in run[1:]:
            idx = add_point(p)
            instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

        # pen up
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
    lines.append("///DATE 2026/04/22 19:50")
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
        print("Usage: python TimYas_black_runs.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)

    runs = extract_black_runs(
        chosen_mask,
        row_step=ROW_STEP,
        min_run_length=MIN_RUN_LENGTH
    )

    save_run_debug_image(chosen_mask, runs)
    mapped_runs, mapping = runs_to_template_pulses(runs)

    save_point_mapping(mapping)
    write_jbi_black_runs(mapped_runs, filename=JOB_FILE, job_name=JOB_NAME)

    print("\\nDone.")
    print(f"Found {len(runs)} black runs.")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - point_mapping_runs.txt")
    print(f"  - {JOB_FILE}")
    print("\\nBehavior:")
    print("  - black area -> pen down")
    print("  - white gap -> pen up")
    print("  - fills dark regions with scanline strokes")
    print(f"  - ROW_STEP: {ROW_STEP}")
    print(f"  - MIN_RUN_LENGTH: {MIN_RUN_LENGTH}")

if __name__ == "__main__":
    main()