import cv2
import numpy as np
import sys
import json
from pathlib import Path

# =========================================================
# LOAD SKIN CORNERS FROM skin_corners.json
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


TL, TR, BR, BL, CENTER = load_skin_corners()
APPROACH_POINT = CENTER.copy()

# =========================================================
# INK DIP POSITIONS
# =========================================================

INK_HOVER = [-41004, 59868,  1099, 4900, -34257, 16705]
INK_DIP   = [-41004, 64596,  1380, 5213, -31659, 16463]

# Raise U before traveling TO ink to clear table
INK_PRELIFT_U    = 3000
# Raise U after dipping to clear platform on return
INK_CLEAR_LIFT_U = 8000

# Redip every 50 seconds
REDIP_INTERVAL_SEC = 50.0

# =========================================================
# DRAWING SIZE CONTROL
# Set exact width and height of the tattoo in mm.
# The image will be scaled to fit this box while
# preserving aspect ratio to avoid distortion.
# Skin is approximately 159mm wide x 141mm tall.
# =========================================================

TATTOO_WIDTH_MM  = 70.0   # desired tattoo width in mm
TATTOO_HEIGHT_MM = 70.0   # desired tattoo height in mm

SKIN_WIDTH_MM  = 159.0    # physical skin width in mm
SKIN_HEIGHT_MM = 141.0    # physical skin height in mm

# These become the UV scale factors
# image fills this fraction of the skin
SCALE_U = TATTOO_WIDTH_MM  / SKIN_WIDTH_MM
SCALE_V = TATTOO_HEIGHT_MM / SKIN_HEIGHT_MM

# =========================================================
# TUNING
# =========================================================

JOB_NAME           = "CONTOUR2"
JOB_FILE           = "CONTOUR2.JBI"
MOVEJ_SPEED        = 0.78
MOVL_SPEED         = 8.0
MOVL_SPEED_INK     = 11.7

# J3 lift for pen-up between contours
LIFT_J3            = 1400

# APPROACH_LIFT_J3 is a BIGGER lift used when approaching
# the skin from above — prevents dragging lines
# Must be larger than LIFT_J3
APPROACH_LIFT_J3   = 2000

MIN_CONTOUR_AREA   = 30
POINTS_PER_CONTOUR = 60
AIR_ONLY_MODE      = False

AVG_POINT_SPACING_MM   = 2.5
SECONDS_PER_DRAW_POINT = AVG_POINT_SPACING_MM / MOVL_SPEED

# =========================================================
# IMAGE CLEANUP
# =========================================================

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(binary,
                                          connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def method_v1(cropped):
    gray      = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
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
    gray         = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_blur    = cv2.GaussianBlur(gray, (3, 3), 0)
    background   = cv2.GaussianBlur(gray_blur, (31, 31), 0)
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    cv2.putText(img2, "2: Thin/Sketch Mode", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 0, 0), 2)
    return np.hstack((img1, img2))


def choose_trace_mask(img):
    print("Select the design area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select Design Area", img,
                         showCrosshair=True,
                         fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    cropped = img[y:y+h, x:x+w] \
              if (w > 0 and h > 0) else img.copy()

    final1  = method_v1(cropped)
    final2  = method_v1b(cropped)
    compare = stack_for_display(final1, final2)
    cv2.imshow("Choose Best Result: Press 1 or 2", compare)
    print("Press 1 for thick-line mode, "
          "2 for thin-sketch mode.")
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
        idx     = np.searchsorted(cumulative, t,
                                   side="right") - 1
        idx     = min(max(idx, 0), len(seg_lengths) - 1)
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
# Preserves image aspect ratio to prevent distortion.
# Image is scaled to fit within TATTOO_WIDTH_MM x
# TATTOO_HEIGHT_MM while maintaining proportions.
# =========================================================

def bilinear_interp(u, v, tl, tr, br, bl):
    top    = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br
    return (1 - v) * top + v * bottom


def all_contours_bounds(contours):
    pts = np.vstack(contours)
    return (np.min(pts[:, 0]), np.max(pts[:, 0]),
            np.min(pts[:, 1]), np.max(pts[:, 1]))


def contours_to_skin_pulses(contours):
    min_x, max_x, min_y, max_y = \
        all_contours_bounds(contours)
    img_w = max(max_x - min_x, 1.0)
    img_h = max(max_y - min_y, 1.0)

    # Preserve aspect ratio — scale image to fit within
    # TATTOO_WIDTH_MM x TATTOO_HEIGHT_MM box
    img_aspect    = img_w / img_h
    tattoo_aspect = TATTOO_WIDTH_MM / TATTOO_HEIGHT_MM

    if img_aspect > tattoo_aspect:
        # Image is wider — fit to width, letterbox height
        actual_scale_u = SCALE_U
        actual_scale_v = SCALE_U / img_aspect * \
                         (SKIN_WIDTH_MM / SKIN_HEIGHT_MM)
    else:
        # Image is taller — fit to height, letterbox width
        actual_scale_v = SCALE_V
        actual_scale_u = SCALE_V * img_aspect * \
                         (SKIN_HEIGHT_MM / SKIN_WIDTH_MM)

    print(f"\nImage aspect ratio: {img_aspect:.3f}")
    print(f"Scale U: {actual_scale_u:.4f}  "
          f"Scale V: {actual_scale_v:.4f}")
    print(f"Effective tattoo size: "
          f"{actual_scale_u * SKIN_WIDTH_MM:.1f}mm x "
          f"{actual_scale_v * SKIN_HEIGHT_MM:.1f}mm")

    mapped_contours = []
    mapping         = []
    gi              = 0

    for ci, contour in enumerate(contours):
        mapped = []
        for pi, (x, y) in enumerate(contour):
            u = (x - min_x) / img_w
            v = (y - min_y) / img_h

            # Scale around center preserving aspect ratio
            u = 0.5 + (u - 0.5) * actual_scale_u
            v = 0.5 + (v - 0.5) * actual_scale_v

            pulse     = bilinear_interp(u, v,
                                        TL, TR, BR, BL)
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
        best_i       = None
        best_rotated = None
        best_dist    = None

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

def save_labeled_contour_image(
        mask, contours,
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


def save_travel_order_image(
        mask, contours,
        output_path="travel_order.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        pts = np.array(contour, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(debug, tuple(pts[i][:2]),
                     tuple(pts[i+1][:2]),
                     (0, 255, 0), 1)
        start = tuple(pts[0][:2])
        cv2.circle(debug, start, 7, (255, 0, 0), -1)
        cv2.putText(debug, str(ci),
                    (start[0]+4, start[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_point_mapping(
        mapping,
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
            f.write(
                f"Global Point {item['global_index']}\n")
            f.write(
                f"  Contour/Point : "
                f"({item['contour_index']}, "
                f"{item['point_index']})\n")
            f.write(
                f"  Image Point   : "
                f"({item['image_x']:.2f}, "
                f"{item['image_y']:.2f})\n")
            f.write(
                f"  UV            : "
                f"({item['u']:.4f}, {item['v']:.4f})\n")
            f.write(
                f"  Pulse         : {item['pulse']}\n\n")
    print(f"Saved: {output_path}")


# =========================================================
# LIFT HELPERS
# Two lift levels:
#   LIFT_J3          — small lift between draw points
#   APPROACH_LIFT_J3 — big lift when approaching skin
#                      from above to prevent drag lines
# =========================================================

def make_lifted_pulse_point(draw_pulse):
    """Small lift — used between contour points."""
    lifted    = list(draw_pulse)
    lifted[2] += LIFT_J3
    return lifted


def make_approach_pulse_point(draw_pulse):
    """
    Big lift — used when approaching skin from above.
    Robot comes straight down to draw point from height
    preventing diagonal drag lines on the skin.
    """
    lifted    = list(draw_pulse)
    lifted[2] += APPROACH_LIFT_J3
    return lifted


# =========================================================
# INK DIP SEQUENCE
# =========================================================

def insert_ink_dip(all_points, instructions):
    """
    Full ink dip sequence:
    prelift → hover → dip → wait 2s → hover → clear lift
    """

    # 0. Pre-lift — raise U before traveling to ink
    prelift_pulse    = list(INK_HOVER)
    prelift_pulse[2] += INK_PRELIFT_U
    prelift_idx = len(all_points)
    all_points.append(prelift_pulse)
    instructions.append(
        f"MOVJ C{prelift_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # 1. Move to hover above ink jar
    hover_idx = len(all_points)
    all_points.append(list(INK_HOVER))
    instructions.append(
        f"MOVJ C{hover_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # 2. Dip into ink
    dip_idx = len(all_points)
    all_points.append(list(INK_DIP))
    instructions.append(
        f"MOVL C{dip_idx:05d} V={MOVL_SPEED_INK:.1f}")

    # 3. Wait 2 seconds while submerged
    instructions.append("TIMER T=2.00")

    # 4. Retract back to hover
    retract_idx = len(all_points)
    all_points.append(list(INK_HOVER))
    instructions.append(
        f"MOVL C{retract_idx:05d} V={MOVL_SPEED_INK:.1f}")

    # 5. Raise higher to clear platform before moving to skin
    clear_pulse    = list(INK_HOVER)
    clear_pulse[2] += INK_CLEAR_LIFT_U
    clear_idx = len(all_points)
    all_points.append(clear_pulse)
    instructions.append(
        f"MOVJ C{clear_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    print(f"  [INK DIP] inserted at point {prelift_idx}")


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

    # ── Initial ink dip ───────────────────────────────────
    print("Inserting initial ink dip...")
    insert_ink_dip(all_points, instructions)

    # ── Approach skin center from high above ─────────────
    approach_high = make_approach_pulse_point(
        APPROACH_POINT.astype(int).tolist())
    approach_high_idx = add_point(approach_high)
    instructions.append(
        f"MOVJ C{approach_high_idx:05d} "
        f"VJ={MOVEJ_SPEED:.2f}")

    # ── Draw contours with timed redips ──────────────────
    elapsed_since_dip = 0.0
    total_dips        = 1
    total_draw_pts    = 0

    for contour in mapped_contours:
        if len(contour) < 2:
            continue

        start = contour[0]
        end   = contour[-1]

        # Use APPROACH lift (big) for coming down to start
        approach_start = make_approach_pulse_point(start)
        # Use regular lift (small) for pen-up between points
        lifted_end     = make_lifted_pulse_point(end)

        if AIR_ONLY_MODE:
            contour = [make_lifted_pulse_point(p)
                       for p in contour]

        # Come straight down onto first point of contour
        # approach_start → start = straight vertical drop
        # This prevents diagonal drag lines
        ap_idx = add_point(approach_start)
        s_idx  = add_point(contour[0])
        instructions.append(
            f"MOVJ C{ap_idx:05d} VJ={MOVEJ_SPEED:.2f}")
        instructions.append(
            f"MOVL C{s_idx:05d} V={MOVL_SPEED:.1f}")

        # Draw each point in the contour
        for p in contour[1:]:

            # Check if time to redip
            elapsed_since_dip += SECONDS_PER_DRAW_POINT
            if elapsed_since_dip >= REDIP_INTERVAL_SEC:

                # Lift straight up before leaving
                current_approach = \
                    make_approach_pulse_point(p)
                lift_idx = add_point(
                    make_lifted_pulse_point(p))
                instructions.append(
                    f"MOVL C{lift_idx:05d} "
                    f"V={MOVL_SPEED:.1f}")

                # Do the ink dip
                insert_ink_dip(all_points, instructions)
                total_dips       += 1
                elapsed_since_dip = 0.0

                # Come straight back down onto resume point
                ap_resume = add_point(
                    make_approach_pulse_point(p))
                resume_idx = add_point(p)
                instructions.append(
                    f"MOVJ C{ap_resume:05d} "
                    f"VJ={MOVEJ_SPEED:.2f}")
                instructions.append(
                    f"MOVL C{resume_idx:05d} "
                    f"V={MOVL_SPEED:.1f}")
            else:
                idx = add_point(p)
                instructions.append(
                    f"MOVL C{idx:05d} "
                    f"V={MOVL_SPEED:.1f}")

            total_draw_pts += 1

        # Lift straight up at end of contour
        le_idx = add_point(lifted_end)
        instructions.append(
            f"MOVL C{le_idx:05d} V={MOVL_SPEED:.1f}")

    # ── Build JBI ─────────────────────────────────────────
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
    print(f"Total points:     {len(all_points)}")
    print(f"Total draw pts:   {total_draw_pts}")
    print(f"Total ink dips:   {total_dips}")
    print(f"Redip interval:   {REDIP_INTERVAL_SEC}s")
    print(f"Tattoo size:      "
          f"{TATTOO_WIDTH_MM}mm x {TATTOO_HEIGHT_MM}mm")
    print(f"Approach lift J3: {APPROACH_LIFT_J3} pulses")
    print(f"Draw lift J3:     {LIFT_J3} pulses")
    print(f"Est. draw time:   "
          f"{total_draw_pts * SECONDS_PER_DRAW_POINT:.1f}s")


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
    print(f"\nSettings:")
    print(f"  Tattoo size:      "
          f"{TATTOO_WIDTH_MM}mm x {TATTOO_HEIGHT_MM}mm")
    print(f"  AIR_ONLY_MODE:    {AIR_ONLY_MODE}")
    print(f"  Points/contour:   {points_per_contour}")
    print(f"  Approach lift J3: {APPROACH_LIFT_J3}")
    print(f"  Draw lift J3:     {LIFT_J3}")
    print(f"  Redip interval:   {REDIP_INTERVAL_SEC}s")
    print(f"  Draw speed:       {MOVL_SPEED} mm/s")
    print(f"  Ink dip speed:    {MOVL_SPEED_INK} mm/s")
    print(f"  Pre-lift U:       +{INK_PRELIFT_U} pulses")
    print(f"  Clear lift U:     +{INK_CLEAR_LIFT_U} pulses")


if __name__ == "__main__":
    main()