import cv2
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime

# =========================================================
# FILES
# =========================================================

WORKSPACE_JSON      = "fake_skin_coordinates_robot.json"
OUTPUT_CONTOUR_JSON = "mapped_contours_robot_xy.json"
OUTPUT_MAPPING_TXT  = "point_mapping_robot_xy.txt"

JOB_NAME = "CONTOUR2"
JOB_FILE = "CONTOUR2.JBI"

# =========================================================
# REFERENCE POSE
# Jog the robot to your draw position on the pendant.
# Read BOTH Cartesian XYZ and Pulse counts and paste below.
# Everything is derived from this single reference.
# =========================================================

REF_X_MM =  418.673
REF_Y_MM =  -41.875
REF_Z_MM =   41.642

REF_S = -5186
REF_L =  147410
REF_U = -9990
REF_R =  1217
REF_B = -52936
REF_T =  1935

# =========================================================
# DRAW / SAFE Z  (mm, same Cartesian frame as REF_Z_MM)
# =========================================================

DRAW_Z_MM = -159.668
SAFE_Z_MM = -129.668

# =========================================================
# Z-AXIS PULSE SCALE
# How many pulses L and U change per 1mm of Z travel.
# Defaults are typical for a Yaskawa arm at mid-reach.
# If Z motion looks wrong, tune these two values.
# For best accuracy: teach a second point at a different Z,
# read its L and U pulses, then set USE_SECOND_Z_REF = True.
# =========================================================

DL_PER_MM_Z = 500.0    # L pulses per mm of Z
DU_PER_MM_Z = -500.0   # U pulses per mm of Z

USE_SECOND_Z_REF = False
REF2_Z_MM = 0.0
REF2_L    = 0
REF2_U    = 0

if USE_SECOND_Z_REF and (REF2_Z_MM != REF_Z_MM):
    dz = REF2_Z_MM - REF_Z_MM
    DL_PER_MM_Z = (REF2_L - REF_L) / dz
    DU_PER_MM_Z = (REF2_U - REF_U) / dz

# =========================================================
# XY PULSE SCALE  (computed from workspace, tunable here)
# Pulses per mm for each Cartesian axis mapped to S and L.
# Typical Yaskawa at ~400mm reach: 150 pulses/mm.
# =========================================================

PULSES_PER_MM_X_S = 0.0
PULSES_PER_MM_X_L = 150.0
PULSES_PER_MM_Y_S = 150.0
PULSES_PER_MM_Y_L = 0.0

# =========================================================
# MOTION SETTINGS
# =========================================================

MOVEJ_SPEED      = 0.78
MOVEJ_SPEED_DRAW = 0.10

# =========================================================
# TUNING
# =========================================================

MIN_CONTOUR_AREA   = 30
POINTS_PER_CONTOUR = 60

# =========================================================
# PULSE CONVERSION
# =========================================================

def compute_pulse_scale(tl, tr, bl):
    global PULSES_PER_MM_X_S, PULSES_PER_MM_X_L
    global PULSES_PER_MM_Y_S, PULSES_PER_MM_Y_L

    x_span_mm = float(np.linalg.norm(tr - tl))
    y_span_mm = float(np.linalg.norm(bl - tl))

    x_dir = (tr - tl) / max(x_span_mm, 1.0)
    y_dir = (bl - tl) / max(y_span_mm, 1.0)

    PULSES_PER_MM_X_S = 150.0 * float(x_dir[1])
    PULSES_PER_MM_X_L = 150.0 * float(x_dir[0])
    PULSES_PER_MM_Y_S = 150.0 * float(y_dir[1])
    PULSES_PER_MM_Y_L = 150.0 * float(y_dir[0])

    print(f"\nWorkspace span : X={x_span_mm:.1f}mm  Y={y_span_mm:.1f}mm")
    print(f"Pulse scale    : X->dS={PULSES_PER_MM_X_S:.2f} dL={PULSES_PER_MM_X_L:.2f}")
    print(f"               : Y->dS={PULSES_PER_MM_Y_S:.2f} dL={PULSES_PER_MM_Y_L:.2f}")
    print(f"               : Z->dL={DL_PER_MM_Z:.2f} dU={DU_PER_MM_Z:.2f}")
    print("NOTE: If motion is scaled wrong, adjust PULSES_PER_MM_* at the top.")


def xyz_to_pulse(x_mm, y_mm, z_mm):
    """
    Convert Cartesian XYZ (mm) to pulse counts using a
    linearised offset from the taught reference pose.
    R, B, T (wrist) are kept fixed — orientation unchanged.
    """
    dx = x_mm - REF_X_MM
    dy = y_mm - REF_Y_MM
    dz = z_mm - REF_Z_MM

    s = int(round(REF_S + dx * PULSES_PER_MM_X_S + dy * PULSES_PER_MM_Y_S))
    l = int(round(REF_L + dx * PULSES_PER_MM_X_L + dy * PULSES_PER_MM_Y_L + dz * DL_PER_MM_Z))
    u = int(round(REF_U + dz * DU_PER_MM_Z))
    r = REF_R
    b = REF_B
    t = REF_T

    return [s, l, u, r, b, t]


def format_pulse_point(pulse):
    return ",".join(str(v) for v in pulse)


# =========================================================
# IMAGE CLEANUP
# =========================================================

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def method_v1(cropped):
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = remove_small_components(cleaned, min_area=20)
    final_mask = np.full_like(cleaned, 255)
    final_mask[cleaned > 0] = 0
    return final_mask


def method_v1b(cropped):
    gray       = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.GaussianBlur(gray_blur, (31, 31), 0)
    ink_response = cv2.subtract(background, gray_blur)
    ink_response = cv2.normalize(ink_response, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(ink_response, 40, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=1)
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
        return cv2.copyMakeBorder(img, 0, pad, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    img1 = pad_to_height(img1, h)
    img2 = pad_to_height(img2, h)
    cv2.putText(img1, "1: Thick/Fill Mode",  (10, 25),
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
    cropped = img[y:y + h, x:x + w] if (w > 0 and h > 0) else img.copy()

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

    diffs       = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative  = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total       = cumulative[-1]

    if total <= 0:
        return points[:1]

    targets = np.linspace(0, total, target_points, endpoint=False)
    sampled = []

    for t in targets:
        idx     = np.searchsorted(cumulative, t, side="right") - 1
        idx     = min(max(idx, 0), len(seg_lengths) - 1)
        start   = points[idx]
        end     = points[idx + 1]
        seg_len = seg_lengths[idx]
        p       = start if seg_len == 0 else (
                      start + ((t - cumulative[idx]) / seg_len) * (end - start)
                  )
        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)
    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])
    return sampled


def extract_contours(mask, min_area=30, points_per_contour=60):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contours found.")

    kept = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            pts = contour[:, 0, :].astype(np.float32)
            pts = resample_closed_contour(pts, target_points=points_per_contour)
            if len(pts) >= 3:
                kept.append(pts)

    if not kept:
        raise RuntimeError("Contours were found but all were too small after filtering.")
    return kept


# =========================================================
# LOAD WORKSPACE JSON
# =========================================================

def load_workspace_from_json(json_path=WORKSPACE_JSON):
    with open(json_path, "r") as f:
        data = json.load(f)

    corners = data["corners_robot_xy_mm"]
    tl = np.array(corners["TL"], dtype=np.float64)
    tr = np.array(corners["TR"], dtype=np.float64)
    br = np.array(corners["BR"], dtype=np.float64)
    bl = np.array(corners["BL"], dtype=np.float64)

    print("\nLoaded fake skin workspace:")
    print(f"  TL: {tl.tolist()}")
    print(f"  TR: {tr.tolist()}")
    print(f"  BR: {br.tolist()}")
    print(f"  BL: {bl.tolist()}")

    return tl, tr, br, bl, data


# =========================================================
# MAP CONTOURS TO WORKSPACE XY
# =========================================================

def bilinear_interp_xy(u, v, tl, tr, br, bl):
    top    = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br
    return (1 - v) * top + v * bottom


def all_contours_bounds(contours):
    pts   = np.vstack(contours)
    return (np.min(pts[:, 0]), np.max(pts[:, 0]),
            np.min(pts[:, 1]), np.max(pts[:, 1]))


def contours_to_robot_xy(contours, tl, tr, br, bl):
    min_x, max_x, min_y, max_y = all_contours_bounds(contours)
    width  = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_contours = []
    mapping         = []
    global_index    = 0

    for contour_index, contour in enumerate(contours):
        mapped = []
        for point_index, (x, y) in enumerate(contour):
            u        = (x - min_x) / width
            v        = (y - min_y) / height
            robot_xy = bilinear_interp_xy(u, v, tl, tr, br, bl)

            mapped.append({"x_mm": float(robot_xy[0]),
                           "y_mm": float(robot_xy[1])})
            mapping.append({
                "global_index":  global_index,
                "contour_index": contour_index,
                "point_index":   point_index,
                "image_x":       float(x),
                "image_y":       float(y),
                "u":             float(u),
                "v":             float(v),
                "robot_x_mm":    float(robot_xy[0]),
                "robot_y_mm":    float(robot_xy[1])
            })
            global_index += 1
        mapped_contours.append(mapped)

    return mapped_contours, mapping


# =========================================================
# DEBUG IMAGES
# =========================================================

def save_labeled_contour_image(mask, contours,
                               output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        for pi, p in enumerate(contour):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(debug, f"{ci}:{pi}", (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_travel_order_image(mask, contours,
                            output_path="travel_order.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        pts = np.array(contour, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(debug, tuple(pts[i][:2]), tuple(pts[i + 1][:2]),
                     (0, 255, 0), 1)
        start = tuple(pts[0][:2])
        cv2.circle(debug, start, 7, (255, 0, 0), -1)
        cv2.putText(debug, str(ci), (start[0] + 4, start[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_point_mapping(mapping, output_path=OUTPUT_MAPPING_TXT):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping: Design Image -> Robot XY Workspace\n")
        f.write("=" * 78 + "\n\n")
        for item in mapping:
            f.write(f"Global Point {item['global_index']}\n")
            f.write(f"  Contour/Point : ({item['contour_index']}, {item['point_index']})\n")
            f.write(f"  Image Point   : ({item['image_x']:.2f}, {item['image_y']:.2f})\n")
            f.write(f"  Workspace UV  : ({item['u']:.4f}, {item['v']:.4f})\n")
            f.write(f"  Robot XY mm   : X={item['robot_x_mm']:.3f}, Y={item['robot_y_mm']:.3f}\n\n")
    print(f"Saved: {output_path}")


def save_robot_xy_json(mapped_contours, mapping, workspace_data,
                       output_path=OUTPUT_CONTOUR_JSON):
    output = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "method":           "XYZ to Pulse via linearised Jacobian from reference pose",
        "workspace_source": WORKSPACE_JSON,
        "reference_pose": {
            "x_mm": REF_X_MM, "y_mm": REF_Y_MM, "z_mm": REF_Z_MM,
            "S": REF_S, "L": REF_L, "U": REF_U,
            "R": REF_R, "B": REF_B, "T": REF_T
        },
        "draw_z_mm":            DRAW_Z_MM,
        "safe_z_mm":            SAFE_Z_MM,
        "fake_skin_workspace":  workspace_data,
        "contours_robot_xy_mm": mapped_contours,
        "flat_point_mapping":   mapping
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved: {output_path}")


# =========================================================
# PULSE JBI WRITER
# =========================================================

def write_jbi_pulse(mapped_contours, filename=JOB_FILE, job_name=JOB_NAME):

    all_points   = []
    instructions = []

    def add_point(pulse):
        idx = len(all_points)
        all_points.append(pulse)
        return idx

    # Approach: safe Z above first contour start
    first        = mapped_contours[0][0]
    approach_idx = add_point(xyz_to_pulse(first["x_mm"], first["y_mm"], SAFE_Z_MM))
    instructions.append(f"MOVJ C{approach_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    for contour in mapped_contours:
        if len(contour) < 2:
            continue

        p0 = contour[0]

        # Safe Z above contour start
        ls_idx = add_point(xyz_to_pulse(p0["x_mm"], p0["y_mm"], SAFE_Z_MM))
        instructions.append(f"MOVJ C{ls_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        # Plunge to draw Z (slow)
        s_idx = add_point(xyz_to_pulse(p0["x_mm"], p0["y_mm"], DRAW_Z_MM))
        instructions.append(f"MOVJ C{s_idx:05d} VJ={MOVEJ_SPEED_DRAW:.2f}")

        # Draw contour
        for p in contour[1:]:
            idx = add_point(xyz_to_pulse(p["x_mm"], p["y_mm"], DRAW_Z_MM))
            instructions.append(f"MOVJ C{idx:05d} VJ={MOVEJ_SPEED_DRAW:.2f}")

        # Lift after contour
        pe     = contour[-1]
        le_idx = add_point(xyz_to_pulse(pe["x_mm"], pe["y_mm"], SAFE_Z_MM))
        instructions.append(f"MOVJ C{le_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # Final safe retreat
    last      = mapped_contours[-1][-1]
    final_idx = add_point(xyz_to_pulse(last["x_mm"], last["y_mm"], SAFE_Z_MM))
    instructions.append(f"MOVJ C{final_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # Build JBI
    lines = [
        "/JOB",
        f"//NAME {job_name}",
        "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0",
        "///POSTYPE PULSE",
        "///PULSE",
    ]

    for i, pulse in enumerate(all_points):
        lines.append(f"C{i:05d}={format_pulse_point(pulse)}")

    lines += [
        "//INST",
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW",
        "///GROUP1 RB1",
        "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI file     : {filename}")
    print(f"Total points       : {len(all_points)}")
    print(f"Format             : PULSE")
    print(f"Reference XYZ      : X={REF_X_MM}, Y={REF_Y_MM}, Z={REF_Z_MM}")
    print(f"Reference Pulses   : S={REF_S}, L={REF_L}, U={REF_U}, R={REF_R}, B={REF_B}, T={REF_T}")
    print(f"Draw Z             : {DRAW_Z_MM} mm")
    print(f"Safe Z             : {SAFE_Z_MM} mm")
    print("\n*** TEST IN AIR AT LOW SPEED FIRST ***")
    print("If XY motion is wrong -> adjust PULSES_PER_MM_* at the top of the script.")
    print("If Z motion is wrong  -> adjust DL_PER_MM_Z and DU_PER_MM_Z.")
    print("For best Z accuracy   -> teach a 2nd Z point and set USE_SECOND_Z_REF=True.")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_to_skin_workspace_jbi.py "
              "<image_path> [points_per_contour]")
        sys.exit(1)

    image_path         = Path(sys.argv[1])
    points_per_contour = int(sys.argv[2]) if len(sys.argv) >= 3 else POINTS_PER_CONTOUR

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tl, tr, br, bl, workspace_data = load_workspace_from_json(WORKSPACE_JSON)

    compute_pulse_scale(tl, tr, bl)

    chosen_mask = choose_trace_mask(img)

    contours = extract_contours(
        chosen_mask,
        min_area=MIN_CONTOUR_AREA,
        points_per_contour=points_per_contour
    )

    save_labeled_contour_image(chosen_mask, contours)
    save_travel_order_image(chosen_mask, contours)

    mapped_contours, mapping = contours_to_robot_xy(contours, tl, tr, br, bl)

    save_robot_xy_json(mapped_contours, mapping, workspace_data)
    save_point_mapping(mapping)

    write_jbi_pulse(mapped_contours, filename=JOB_FILE, job_name=JOB_NAME)

    print("\nDone.")
    print(f"Found {len(contours)} contour(s).")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - travel_order.png")
    print(f"  - {OUTPUT_MAPPING_TXT}")
    print(f"  - {OUTPUT_CONTOUR_JSON}")
    print(f"  - {JOB_FILE}")


if __name__ == "__main__":
    main()