import cv2
import numpy as np
import sys
import json
from pathlib import Path

# =========================================================
# LOAD SKIN CORNERS FROM skin_corners.json
# The 4 corners define where the image will be drawn.
# The image is mapped to fit inside these 4 corners.
# =========================================================

def load_skin_corners(json_path="skin_corners.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    corners = {}
    for c in data["corners"]:
        corners[c["corner"]] = np.array(c["pulse"],
                                         dtype=np.float64)

    TL = corners["TL"]
    TR = corners["TR"]
    BR = corners["BR"]
    BL = corners["BL"]

    center = (TL + TR + BR + BL) / 4.0

    print("\nLoaded skin corners from skin_corners.json:")
    print(f"  TL: {TL.astype(int).tolist()}")
    print(f"  TR: {TR.astype(int).tolist()}")
    print(f"  BR: {BR.astype(int).tolist()}")
    print(f"  BL: {BL.astype(int).tolist()}")
    print(f"  Center: {center.astype(int).tolist()}")

    return TL, TR, BR, BL, center


# Load corners
TL, TR, BR, BL, CENTER = load_skin_corners()

# Approach point is the center of the skin
APPROACH_POINT = CENTER.copy()

# =========================================================
# TUNING
# =========================================================

JOB_NAME = "CONTOUR2"
JOB_FILE = "CONTOUR2.JBI"

MOVEJ_SPEED      = 0.78
MOVL_SPEED       = 8.0
LIFT_J3          = 1400
MIN_CONTOUR_AREA = 30
POINTS_PER_CONTOUR = 60
AIR_ONLY_MODE    = False

# ── Drawing box size ──────────────────────────────────────
# The robot will draw the image inside a box this size,
# centered on the skin. Change DRAW_BOX_MM to resize.
# Physical skin size is measured from skin_corners.json
# cartesian values (~159mm wide x ~141mm tall).
DRAW_BOX_MM   = 70          # box width AND height in mm
SKIN_WIDTH_MM  = 141.2     # measured from cartesian corners
SKIN_HEIGHT_MM = 141.2      # measured from cartesian corners

# Fraction of skin the box occupies (auto-calculated)
SCALE_U = DRAW_BOX_MM / SKIN_WIDTH_MM   # ~0.439
SCALE_V = DRAW_BOX_MM / SKIN_HEIGHT_MM  # ~0.496


# =========================================================
# IMAGE CLEANUP
# =========================================================

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def method_v1(cropped):
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 90, 255,
                               cv2.THRESH_BINARY_INV)
    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
                                kernel, iterations=1)
    cleaned = remove_small_components(cleaned, min_area=20)
    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0
    return final_mask


def method_v1b(cropped):
    gray       = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.GaussianBlur(gray_blur, (31, 31), 0)
    ink_response = cv2.subtract(background, gray_blur)
    ink_response = cv2.normalize(ink_response, None, 0, 255,
                                  cv2.NORM_MINMAX)
    _, binary = cv2.threshold(ink_response, 40, 255,
                               cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
                                kernel, iterations=1)
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
        return cv2.copyMakeBorder(img, 0, pad, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))

    img1 = pad_to_height(img1, h)
    img2 = pad_to_height(img2, h)
    cv2.putText(img1, "1: Thick/Fill Mode", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img2, "2: Thin/Sketch Mode", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return np.hstack((img1, img2))


def choose_trace_mask(img):
    print("Select the design area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select Design Area", img,
                         showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    cropped = img[y:y+h, x:x+w] if (w > 0 and h > 0) \
              else img.copy()

    final1 = method_v1(cropped)
    final2 = method_v1b(cropped)

    compare = stack_for_display(final1, final2)
    cv2.imshow("Choose Best Result: Press 1 or 2", compare)
    print("Press 1 for thick-line mode, 2 for thin-sketch mode.")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord("1"):
        chosen = final1
        print("Chose Method 1")
    elif key == ord("2"):
        chosen = final2
        print("Chose Method 2")
    else:
        raise RuntimeError("No valid choice. Press 1 or 2.")

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

    diffs       = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative  = np.concatenate([[0.0],
                                   np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 0:
        return points[:1]

    targets = np.linspace(0, total, target_points,
                           endpoint=False)
    sampled = []
    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(idx, 0), len(seg_lengths) - 1)
        start   = points[idx]
        end     = points[idx + 1]
        seg_len = seg_lengths[idx]
        p = start if seg_len == 0 else (
            start + ((t - cumulative[idx]) / seg_len)
            * (end - start)
        )
        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)
    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])
    return sampled


def extract_contours(mask, min_area=30,
                     points_per_contour=60):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours found.")

    kept = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            pts = contour[:, 0, :].astype(np.float32)
            pts = resample_closed_contour(
                pts, target_points=points_per_contour)
            if len(pts) >= 3:
                kept.append(pts)

    if not kept:
        raise RuntimeError(
            "All contours too small after filtering.")
    return kept


# =========================================================
# MAP CONTOURS TO SKIN CORNERS
# Image pixel coordinates are mapped into the
# 4 pulse corners from skin_corners.json using
# bilinear interpolation.
# u=0 left→u=1 right  (image x)
# v=0 top →v=1 bottom (image y)
# Corners: TL(u=0,v=0) TR(u=1,v=0) BR(u=1,v=1) BL(u=0,v=1)
# =========================================================

def bilinear_interp(u, v, tl, tr, br, bl):
    """
    Bilinear interpolation across 4 corners.
    TL=top-left, TR=top-right, BR=bottom-right, BL=bottom-left
    """
    top    = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br
    return (1 - v) * top + v * bottom


def all_contours_bounds(contours):
    pts = np.vstack(contours)
    return (np.min(pts[:, 0]), np.max(pts[:, 0]),
            np.min(pts[:, 1]), np.max(pts[:, 1]))


def contours_to_skin_pulses(contours):
    """
    Map image contour points into the skin corner
    pulse space using bilinear interpolation.
    The image fills the area defined by TL/TR/BR/BL.
    """
    min_x, max_x, min_y, max_y = all_contours_bounds(contours)
    width  = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_contours = []
    mapping         = []
    gi              = 0

    for ci, contour in enumerate(contours):
        mapped = []
        for pi, (x, y) in enumerate(contour):
            # Normalize to [0,1] within image bounds
            u = (x - min_x) / width
            v = (y - min_y) / height

            # ── Fit into DRAW_BOX_MM x DRAW_BOX_MM centered box ──
            # Shrink around center (0.5, 0.5) so drawing stays centered.
            u = 0.5 + (u - 0.5) * SCALE_U
            v = 0.5 + (v - 0.5) * SCALE_V

            # Map to pulse space using skin corners
            pulse = bilinear_interp(u, v, TL, TR, BR, BL)
            pulse_int = [int(round(p)) for p in pulse]

            mapped.append(pulse_int)
            mapping.append({
                "global_index":  gi,
                "contour_index": ci,
                "point_index":   pi,
                "image_x":       float(x),
                "image_y":       float(y),
                "u":             float(u),
                "v":             float(v),
                "pulse":         pulse_int,
            })
            gi += 1

        mapped_contours.append(mapped)

    return mapped_contours, mapping


# =========================================================
# CONTOUR REORDERING
# =========================================================

def rotate_contour_to_best_start(contour, ref_point):
    if len(contour) <= 2:
        return contour

    pts    = np.array(contour, dtype=np.int32)
    closed = np.array_equal(pts[0], pts[-1])
    core   = pts[:-1] if closed else pts

    ref  = np.array(ref_point, dtype=np.float32)
    d2   = np.sum((core.astype(np.float32) - ref) ** 2,
                   axis=1)
    best = int(np.argmin(d2))

    rotated = np.vstack([core[best:], core[:best]])
    rotated = np.vstack([rotated, rotated[0]])
    return rotated.tolist()


def reorder_contours_nearest(mapped_contours):
    remaining = [list(map(list, c))
                 for c in mapped_contours]
    if not remaining:
        return []

    ordered     = []
    current_ref = APPROACH_POINT.astype(np.int32).tolist()

    while remaining:
        best_i      = None
        best_rotated = None
        best_dist   = None

        for i, contour in enumerate(remaining):
            rotated = rotate_contour_to_best_start(
                contour, current_ref)
            start = np.array(rotated[0], dtype=np.float32)
            ref   = np.array(current_ref, dtype=np.float32)
            dist  = float(np.sum((start - ref) ** 2))

            if best_dist is None or dist < best_dist:
                best_dist    = dist
                best_i       = i
                best_rotated = rotated

        ordered.append(best_rotated)
        current_ref = best_rotated[-1]
        remaining.pop(best_i)

    return ordered


# =========================================================
# DEBUG OUTPUTS
# =========================================================

def save_labeled_contour_image(mask, contours,
        output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        for pi, p in enumerate(contour):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(debug, f"{ci}:{pi}", (x+3, y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0, 0, 255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_travel_order_image(mask, contours,
        output_path="travel_order.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        pts = np.array(contour, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(debug, tuple(pts[i][:2]),
                     tuple(pts[i+1][:2]), (0, 255, 0), 1)
        start = tuple(pts[0][:2])
        cv2.circle(debug, start, 7, (255, 0, 0), -1)
        cv2.putText(debug, str(ci), (start[0]+4, start[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_point_mapping(mapping,
        output_path="point_mapping_contours.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping: Image -> Skin Pulse Space\n")
        f.write("=" * 78 + "\n\n")
        f.write("Skin corners used:\n")
        f.write(f"  TL: {TL.astype(int).tolist()}\n")
        f.write(f"  TR: {TR.astype(int).tolist()}\n")
        f.write(f"  BR: {BR.astype(int).tolist()}\n")
        f.write(f"  BL: {BL.astype(int).tolist()}\n\n")
        for item in mapping:
            f.write(f"Global Point {item['global_index']}\n")
            f.write(f"  Contour/Point : "
                    f"({item['contour_index']}, "
                    f"{item['point_index']})\n")
            f.write(f"  Image Point   : "
                    f"({item['image_x']:.2f}, "
                    f"{item['image_y']:.2f})\n")
            f.write(f"  UV            : "
                    f"({item['u']:.4f}, {item['v']:.4f})\n")
            f.write(f"  Pulse         : {item['pulse']}\n\n")
    print(f"Saved: {output_path}")


# =========================================================
# LIFT / PEN UP
# =========================================================

def make_lifted_pulse_point(draw_pulse):
    lifted    = list(draw_pulse)
    lifted[2] += LIFT_J3
    return lifted


# =========================================================
# JBI WRITER
# =========================================================

def write_jbi_contour_trace(mapped_contours,
                             filename=JOB_FILE,
                             job_name=JOB_NAME):
    all_points   = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append(p)
        return idx

    # Start at skin center
    approach = APPROACH_POINT.astype(int).tolist()
    if AIR_ONLY_MODE:
        approach = make_lifted_pulse_point(approach)

    approach_idx = add_point(approach)
    instructions.append(
        f"MOVJ C{approach_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    for contour in mapped_contours:
        if len(contour) < 2:
            continue

        start = contour[0]
        end   = contour[-1]

        lifted_start = make_lifted_pulse_point(start)
        lifted_end   = make_lifted_pulse_point(end)

        if AIR_ONLY_MODE:
            contour = [make_lifted_pulse_point(p)
                       for p in contour]

        ls_idx = add_point(lifted_start)
        s_idx  = add_point(contour[0])
        instructions.append(
            f"MOVJ C{ls_idx:05d} VJ={MOVEJ_SPEED:.2f}")
        instructions.append(
            f"MOVJ C{s_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        for p in contour[1:]:
            idx = add_point(p)
            instructions.append(
                f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

        le_idx = add_point(lifted_end)
        instructions.append(
            f"MOVJ C{le_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    lines = [
        "/JOB",
        f"//NAME {job_name}",
        "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0",
        "///POSTYPE PULSE",
        "///PULSE",
    ]
    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}="
            f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")
    lines += [
        "//INST",
        f"///DATE 2026/04/29 10:00",
        "///ATTR SC,RW",
        "///GROUP1 RB1",
        "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8",
              newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI: {filename}")
    print(f"Total points: {len(all_points)}")
    print(f"Skin corners used:")
    print(f"  TL: {TL.astype(int).tolist()}")
    print(f"  TR: {TR.astype(int).tolist()}")
    print(f"  BR: {BR.astype(int).tolist()}")
    print(f"  BL: {BL.astype(int).tolist()}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> "
              "[points_per_contour]")
        sys.exit(1)

    image_path         = Path(sys.argv[1])
    points_per_contour = int(sys.argv[2]) \
                         if len(sys.argv) >= 3 \
                         else POINTS_PER_CONTOUR

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)

    contours = extract_contours(
        chosen_mask,
        min_area=MIN_CONTOUR_AREA,
        points_per_contour=points_per_contour,
    )

    save_labeled_contour_image(chosen_mask, contours)

    mapped_contours, mapping = contours_to_skin_pulses(
        contours)
    ordered_contours = reorder_contours_nearest(
        mapped_contours)

    save_travel_order_image(chosen_mask, ordered_contours)
    save_point_mapping(mapping)
    write_jbi_contour_trace(ordered_contours,
                             filename=JOB_FILE,
                             job_name=JOB_NAME)

    print(f"\nDone. Found {len(contours)} contour(s).")
    print("Saved:")
    print("  chosen_mask.png")
    print("  sampled_points_labeled.png")
    print("  travel_order.png")
    print("  point_mapping_contours.txt")
    print(f"  {JOB_FILE}")
    print(f"\nAIR_ONLY_MODE: {AIR_ONLY_MODE}")
    print(f"Points per contour: {points_per_contour}")
    print(f"Lift J3: {LIFT_J3}")


if __name__ == "__main__":
    main()