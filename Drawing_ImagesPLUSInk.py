import cv2
import numpy as np
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# =========================================================
# 9-POINT CALIBRATION GRID
# =========================================================

CALIBRATION_GRID = [
    (266.281, -103.412, -157.526, -19959,  17347, -46618,  4743, -21772,   6574),
    (264.545,   24.239, -155.017,   5579,  13845, -49514,   479, -21962,  -2952),
    (271.393,  154.316, -156.953,  29100,  21178, -42210, -3340, -23186, -11805),
    (376.842, -108.441, -161.854, -15287,  32940, -28399,  3300, -27100,   5268),
    (375.320,   25.230, -159.374,   4309,  30971, -30912,   590, -26561,  -2412),
    (379.385,  160.264, -158.680,  22794,  35589, -24610, -1949, -28727,  -9681),
    (488.746, -111.483, -165.885, -12411,  49128,  -7449,  2413, -34202,   4504),
    (488.185,   25.841, -165.188,   3338,  47378,  -9679,   603, -33698,  -1941),
    (490.716,  164.271, -164.305,  18692,  51863,  -3631, -1176, -35798,  -8229),
]

GRID_CELLS = [
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (3, 4, 7, 6),
    (4, 5, 8, 7),
]

GRID_PIXELS = [[0, 0]] * 9

# =========================================================
# SKIN CORNER JBI SETTINGS
# =========================================================

SKIN_JOB_FILE     = "SkinDetect.JBI"
SKIN_JOB_NAME     = "SkinDetect"
ROBOT_HOME_PULSE  = [-7969, 21694, -5134, 1465, -52599, 3149]
SKIN_MOVEJ_SPEED  = 0.78
SKIN_MOVL_SPEED   = 11.0
SKIN_CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# =========================================================
# TATTOO JBI SETTINGS
# =========================================================

JOB_NAME = "CONTOUR2"
JOB_FILE = "CONTOUR2.JBI"

LIFT_MM = 10.0
LIFT_10MM_OFFSET = [
    0,      # S
    -1553,  # L
    +209,   # U
    -29,    # R
    -1097,  # B
    +20,    # T
]

INK_HOVER          = [-41004, 59868,  1099, 4900, -34257, 16705]
INK_DIP            = [-41004, 64596,  1380, 5213, -31659, 16463]
INK_PRELIFT_U      = 3000
INK_CLEAR_LIFT_U   = 8000
REDIP_INTERVAL_SEC = 90.0

TATTOO_WIDTH_MM  = 80.0
TATTOO_HEIGHT_MM = 80.0
SKIN_WIDTH_MM    = 159.0
SKIN_HEIGHT_MM   = 141.0
SCALE_U = TATTOO_WIDTH_MM  / SKIN_WIDTH_MM
SCALE_V = TATTOO_HEIGHT_MM / SKIN_HEIGHT_MM

MOVEJ_SPEED        = 1
MOVL_SPEED         = 20.0
MOVL_SPEED_INK     = 50.7
MIN_CONTOUR_AREA   = 5
POINTS_PER_CONTOUR = 60
AIR_ONLY_MODE      = False

REAL_DRAW_SPEED_MM_S   = 8.0
AVG_POINT_SPACING_MM   = 2.5
SECONDS_PER_DRAW_POINT = AVG_POINT_SPACING_MM / REAL_DRAW_SPEED_MM_S

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# =========================================================
# SKIN CORNERS — loaded after detection
# =========================================================

TL = TR = BR = BL = APPROACH_POINT = None


def load_skin_corners(json_path="skin_corners.json"):
    global TL, TR, BR, BL, APPROACH_POINT
    with open(json_path, "r") as f:
        data = json.load(f)

    corners = {}
    for c in data["corners"]:
        corners[c["corner"]] = np.array(
            c["pulse"], dtype=np.float64)

    TL = corners["TL"]
    TR = corners["TR"]
    BR = corners["BR"]
    BL = corners["BL"]
    APPROACH_POINT = (TL + TR + BR + BL) / 4.0

    print("\nLoaded skin corners from skin_corners.json:")
    print(f"  TL: {TL.astype(int).tolist()}")
    print(f"  TR: {TR.astype(int).tolist()}")
    print(f"  BR: {BR.astype(int).tolist()}")
    print(f"  BL: {BL.astype(int).tolist()}")
    print(f"  Center: {APPROACH_POINT.astype(int).tolist()}")


# =========================================================
# PIXEL TO PULSE CONVERSION
# =========================================================

def bilinear_interp_vec(u, v, tl, tr, br, bl):
    top    = (1 - u) * np.array(tl) + u * np.array(tr)
    bottom = (1 - u) * np.array(bl) + u * np.array(br)
    return (1 - v) * top + v * bottom


def find_cell_and_uv(px, py):
    gp        = np.array(GRID_PIXELS, dtype=np.float64)
    best_cell = 0
    best_u    = 0.5
    best_v    = 0.5
    best_dist = float("inf")

    for cell_idx, (tl_i, tr_i, br_i, bl_i) in \
            enumerate(GRID_CELLS):
        src = np.array([gp[tl_i], gp[tr_i],
                        gp[br_i], gp[bl_i]],
                        dtype=np.float32)
        dst = np.array([[0,0],[1,0],[1,1],[0,1]],
                        dtype=np.float32)
        try:
            H   = cv2.getPerspectiveTransform(src, dst)
            pt  = np.array([[[float(px), float(py)]]],
                            dtype=np.float32)
            res = cv2.perspectiveTransform(pt, H)
            u   = float(res[0][0][0])
            v   = float(res[0][0][1])
        except Exception:
            continue

        uc   = max(0.0, min(1.0, u))
        vc   = max(0.0, min(1.0, v))
        dist = (u - uc)**2 + (v - vc)**2

        if dist < best_dist:
            best_dist = dist
            best_cell = cell_idx
            best_u    = uc
            best_v    = vc

    return best_cell, best_u, best_v


def pixel_to_pulse(px, py):
    cell_idx, u, v = find_cell_and_uv(px, py)
    tl_i, tr_i, br_i, bl_i = GRID_CELLS[cell_idx]

    def pulse_of(i):
        p = CALIBRATION_GRID[i]
        return [p[3], p[4], p[5], p[6], p[7], p[8]]

    def cart_of(i):
        p = CALIBRATION_GRID[i]
        return [p[0], p[1], p[2]]

    result_pulse = bilinear_interp_vec(
        u, v,
        pulse_of(tl_i), pulse_of(tr_i),
        pulse_of(br_i), pulse_of(bl_i))

    result_cart = bilinear_interp_vec(
        u, v,
        cart_of(tl_i), cart_of(tr_i),
        cart_of(br_i), cart_of(bl_i))

    pulse = [int(round(x)) for x in result_pulse]
    X = float(result_cart[0])
    Y = float(result_cart[1])
    Z = float(result_cart[2])
    return pulse, X, Y, Z


# =========================================================
# LIFT PULSE
# =========================================================

def lift_pulse(draw_pulse, lift_mm=LIFT_MM):
    scale  = lift_mm / 10.0
    lifted = list(draw_pulse)
    for j in range(6):
        lifted[j] = int(round(
            draw_pulse[j] +
            LIFT_10MM_OFFSET[j] * scale))
    return lifted


# =========================================================
# CAMERA SETUP
# =========================================================

def start_picamera():
    from picamera2 import Picamera2
    import time
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888",
              "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )
    picam2.configure(config)
    picam2.set_controls({"AwbEnable": True,
                          "AeEnable": True})
    picam2.start()
    time.sleep(2)
    return picam2


def get_frame(picam2):
    return picam2.capture_array().copy()


# =========================================================
# SKIN CORNER CLICKING
# =========================================================

clicked_skin_pts = []

def skin_mouse_callback(event, x, y, flags, param):
    global clicked_skin_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_skin_pts) < 4:
            clicked_skin_pts.append([x, y])
            name = SKIN_CORNER_NAMES[
                len(clicked_skin_pts) - 1]
            print(f"  {name}: pixel ({x}, {y})")


def run_skin_corner_click(picam2):
    global clicked_skin_pts
    clicked_skin_pts = []

    print("\n" + "="*55)
    print("MARK SKIN CORNERS")
    print("="*55)
    print("\nClick the 4 corners of the fake skin:")
    print("  1=TOP-LEFT  2=TOP-RIGHT")
    print("  3=BOTTOM-RIGHT  4=BOTTOM-LEFT")
    print("\nR=reset  ENTER=confirm  Q=cancel\n")

    win = "SKIN CORNERS"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, skin_mouse_callback)

    corner_colors = [
        (0,255,255),(255,255,0),
        (0,0,255),(255,0,255),
    ]

    while True:
        frame   = get_frame(picam2)
        display = frame.copy()

        if len(clicked_skin_pts) < 4:
            name = SKIN_CORNER_NAMES[len(clicked_skin_pts)]
            cv2.putText(display,
                f"Click {name} corner  "
                f"({len(clicked_skin_pts)}/4)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,255), 2)
        else:
            cv2.putText(display,
                "All 4 done! Press ENTER to continue.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0,255,0), 2)

        for i, pt in enumerate(clicked_skin_pts):
            cv2.circle(display, tuple(pt), 12,
                       corner_colors[i], -1)
            cv2.putText(display,
                f"{SKIN_CORNER_NAMES[i]} "
                f"({pt[0]},{pt[1]})",
                (pt[0]+10, pt[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, corner_colors[i], 2)

        if len(clicked_skin_pts) == 4:
            pts = np.array(clicked_skin_pts,
                            dtype=np.int32)
            cv2.polylines(display, [pts], True,
                          (0,255,0), 2)

        cv2.putText(display,
            "R=reset | ENTER=confirm | Q=cancel",
            (20, CAMERA_HEIGHT-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255,255,255), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("r"), ord("R")]:
            clicked_skin_pts = []
            print("  Reset.")
        elif key == 13:
            if len(clicked_skin_pts) == 4:
                cv2.destroyWindow(win)
                return clicked_skin_pts
            else:
                print(f"  Need 4 — only "
                      f"{len(clicked_skin_pts)} so far.")
        elif key == ord("q"):
            cv2.destroyWindow(win)
            return None


# =========================================================
# SKIN CORNER JBI WRITER
# =========================================================

def write_skin_jbi(pulse_corners):
    all_points   = []
    instructions = []

    def add_pt(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    instructions.append(
        f"MOVJ C{add_pt(ROBOT_HOME_PULSE):05d} "
        f"VJ={SKIN_MOVEJ_SPEED:.2f}")
    instructions.append(
        f"MOVJ C{add_pt(pulse_corners[0]):05d} "
        f"VJ={SKIN_MOVEJ_SPEED:.2f}")

    for pulse in pulse_corners[1:]:
        instructions.append(
            f"MOVL C{add_pt(pulse):05d} "
            f"V={SKIN_MOVL_SPEED:.1f}")

    instructions.append(
        f"MOVL C{add_pt(pulse_corners[0]):05d} "
        f"V={SKIN_MOVL_SPEED:.1f}")
    instructions.append(
        f"MOVJ C{add_pt(ROBOT_HOME_PULSE):05d} "
        f"VJ={SKIN_MOVEJ_SPEED:.2f}")

    lines = [
        "/JOB", f"//NAME {SKIN_JOB_NAME}", "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0", "///POSTYPE PULSE", "///PULSE",
    ]
    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}="
            f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")
    lines += [
        "//INST",
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW", "///GROUP1 RB1", "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(SKIN_JOB_FILE, "w", encoding="utf-8",
              newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved: {SKIN_JOB_FILE}")


# =========================================================
# PROCESS SKIN CORNERS
# =========================================================

def process_skin_corners(skin_pixels):
    print("\n" + "="*55)
    print("CONVERTING SKIN CORNERS TO PULSE")
    print("="*55)

    pulse_corners = []
    results       = []

    for name, pt in zip(SKIN_CORNER_NAMES, skin_pixels):
        pulse, X, Y, Z = pixel_to_pulse(pt[0], pt[1])
        pulse_corners.append(pulse)
        results.append({
            "corner":    name,
            "pixel":     pt,
            "cartesian": [round(X,3), round(Y,3),
                          round(Z,3)],
            "pulse":     pulse,
        })
        print(f"\n  {name}:")
        print(f"    pixel:     ({pt[0]}, {pt[1]})")
        print(f"    cartesian: "
              f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}")
        print(f"    pulse:     {pulse}")

    write_skin_jbi(pulse_corners)

    out = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"),
        "corners": results,
    }
    with open("skin_corners.json", "w") as f:
        json.dump(out, f, indent=4)

    print(f"\nSaved: skin_corners.json")
    print(f"Saved: {SKIN_JOB_FILE}")


# =========================================================
# IMAGE CLEANUP
# =========================================================

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(
            binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def method_v1(cropped):
    gray      = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blurred, 90, 255,
                               cv2.THRESH_BINARY_INV)
    kernel  = np.ones((3,3), np.uint8)
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
    gray_blur    = cv2.GaussianBlur(gray, (3,3), 0)
    background   = cv2.GaussianBlur(gray_blur, (31,31), 0)
    ink_response = cv2.subtract(background, gray_blur)
    ink_response = cv2.normalize(ink_response, None,
                                  0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(ink_response, 40, 255,
                               cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2,2))
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
            cv2.BORDER_CONSTANT, value=(255,255,255))

    img1 = pad_to_height(img1, h)
    img2 = pad_to_height(img2, h)
    cv2.putText(img1, "1: Thick/Fill Mode", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,0,255), 2)
    cv2.putText(img2, "2: Thin/Sketch Mode", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,0,0), 2)
    return np.hstack((img1, img2))


def choose_trace_mask(img):
    print("Select the design area, "
          "then press ENTER or SPACE.")
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
    cv2.imshow("Choose Best Result: Press 1 or 2",
               compare)
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
        raise RuntimeError(
            "No valid choice. Press 1 or 2.")

    cv2.imwrite("chosen_mask.png", chosen)
    print("Saved chosen_mask.png")
    return chosen


# =========================================================
# CONTOUR TRACING
# Uses arc length as well as area so open curves
# like a smile are not filtered out
# =========================================================

def resample_closed_contour(points, target_points=60):
    points = points.astype(np.float32)
    if len(points) < 2:
        return points
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    diffs       = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative  = np.concatenate(
        [[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 0:
        return points[:1]

    targets = np.linspace(0, total, target_points,
                           endpoint=False)
    sampled = []
    for t in targets:
        idx     = np.searchsorted(cumulative, t,
                                   side="right") - 1
        idx     = min(max(idx, 0),
                      len(seg_lengths) - 1)
        start   = points[idx]
        end     = points[idx+1]
        seg_len = seg_lengths[idx]
        p = start if seg_len == 0 else (
            start + ((t - cumulative[idx]) / seg_len)
            * (end - start))
        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)
    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])
    return sampled


def extract_contours(mask, min_area=5,
                     points_per_contour=60):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(
        inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contours found.")

    kept         = []
    kept_centers = []

    for contour in contours:
        area   = cv2.contourArea(contour)
        length = cv2.arcLength(contour, False)

        # Accept if EITHER area OR arc length is
        # big enough — catches open curves like a
        # smile which have near-zero area
        if area < min_area and length < 10.0:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            # No area — use bounding rect center
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        # Use size = max(area, length) for dedup
        size = max(area, length)

        # Skip near-identical duplicate contours
        is_duplicate = False
        for prev_cx, prev_cy, prev_size in kept_centers:
            dist       = ((cx - prev_cx)**2 +
                          (cy - prev_cy)**2) ** 0.5
            size_ratio = (min(size, prev_size) /
                          max(size, prev_size + 1e-6))
            if dist < 3.0 and size_ratio > 0.9:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        pts = contour[:,0,:].astype(np.float32)
        pts = resample_closed_contour(
            pts, target_points=points_per_contour)

        if len(pts) >= 3:
            kept.append(pts)
            kept_centers.append((cx, cy, size))

    if not kept:
        raise RuntimeError(
            "All contours too small after filtering.")

    print(f"Found {len(kept)} contour(s)")
    return kept


# =========================================================
# MAP CONTOURS TO SKIN CORNERS
# =========================================================

def bilinear_interp(u, v, tl, tr, br, bl):
    top    = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br
    return (1 - v) * top + v * bottom


def all_contours_bounds(contours):
    pts = np.vstack(contours)
    return (np.min(pts[:,0]), np.max(pts[:,0]),
            np.min(pts[:,1]), np.max(pts[:,1]))


def contours_to_skin_pulses(contours):
    min_x, max_x, min_y, max_y = \
        all_contours_bounds(contours)
    img_w = max(max_x - min_x, 1.0)
    img_h = max(max_y - min_y, 1.0)

    img_aspect    = img_w / img_h
    tattoo_aspect = TATTOO_WIDTH_MM / TATTOO_HEIGHT_MM

    if img_aspect > tattoo_aspect:
        actual_scale_u = SCALE_U
        actual_scale_v = SCALE_U / img_aspect * \
                         (SKIN_WIDTH_MM / SKIN_HEIGHT_MM)
    else:
        actual_scale_v = SCALE_V
        actual_scale_u = SCALE_V * img_aspect * \
                         (SKIN_HEIGHT_MM / SKIN_WIDTH_MM)

    print(f"\nImage aspect ratio: {img_aspect:.3f}")
    print(f"Effective tattoo size: "
          f"{actual_scale_u*SKIN_WIDTH_MM:.1f}mm x "
          f"{actual_scale_v*SKIN_HEIGHT_MM:.1f}mm")

    mapped_contours = []
    mapping         = []
    gi              = 0

    for ci, contour in enumerate(contours):
        mapped = []
        for pi, (x, y) in enumerate(contour):
            u = (x - min_x) / img_w
            v = (y - min_y) / img_h
            u = 0.5 + (u - 0.5) * actual_scale_u
            v = 0.5 + (v - 0.5) * actual_scale_v
            pulse = bilinear_interp(
                u, v, TL, TR, BR, BL)
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
    ref    = np.array(ref_point, dtype=np.float32)
    d2     = np.sum(
        (core.astype(np.float32) - ref)**2, axis=1)
    best   = int(np.argmin(d2))
    rotated = np.vstack([core[best:], core[:best]])
    rotated = np.vstack([rotated, rotated[0]])
    return rotated.tolist()


def reorder_contours_nearest(mapped_contours):
    remaining   = [list(map(list, c))
                   for c in mapped_contours]
    ordered     = []
    current_ref = APPROACH_POINT.astype(
        np.int32).tolist()

    while remaining:
        best_i       = None
        best_rotated = None
        best_dist    = None

        for i, contour in enumerate(remaining):
            rotated = rotate_contour_to_best_start(
                contour, current_ref)
            start = np.array(rotated[0],
                              dtype=np.float32)
            ref   = np.array(current_ref,
                              dtype=np.float32)
            dist  = float(np.sum((start - ref)**2))

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
            cv2.circle(debug, (x,y), 4, (0,255,0), -1)
            cv2.putText(debug, f"{ci}:{pi}", (x+3, y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0,0,255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_travel_order_image(
        mask, contours,
        output_path="travel_order.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ci, contour in enumerate(contours):
        pts = np.array(contour, dtype=np.int32)
        for i in range(len(pts)-1):
            cv2.line(debug, tuple(pts[i][:2]),
                     tuple(pts[i+1][:2]),
                     (0,255,0), 1)
        start = tuple(pts[0][:2])
        cv2.circle(debug, start, 7, (255,0,0), -1)
        cv2.putText(debug, str(ci),
                    (start[0]+4, start[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 1)
    cv2.imwrite(output_path, debug)
    print(f"Saved: {output_path}")


def save_point_mapping(
        mapping,
        output_path="point_mapping_contours.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping: Image -> Skin Pulse\n")
        f.write("=" * 78 + "\n\n")
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
# INK DIP SEQUENCE
# =========================================================

def insert_ink_dip(all_points, instructions):
    def add(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    prelift    = list(INK_HOVER)
    prelift[2] += INK_PRELIFT_U
    instructions.append(
        f"MOVJ C{add(prelift):05d} VJ={MOVEJ_SPEED:.2f}")

    instructions.append(
        f"MOVJ C{add(INK_HOVER):05d} VJ={MOVEJ_SPEED:.2f}")

    instructions.append(
        f"MOVL C{add(INK_DIP):05d} V={MOVL_SPEED_INK:.1f}")

    instructions.append("TIMER T=1.00")

    instructions.append(
        f"MOVL C{add(INK_HOVER):05d} V={MOVL_SPEED_INK:.1f}")

    clear    = list(INK_HOVER)
    clear[2] += INK_CLEAR_LIFT_U
    instructions.append(
        f"MOVJ C{add(clear):05d} VJ={MOVEJ_SPEED:.2f}")

    print("  [INK DIP] inserted")


# =========================================================
# JBI WRITER — returns home at end
# =========================================================

def write_jbi_contour_trace(mapped_contours,
                             filename=JOB_FILE,
                             job_name=JOB_NAME):
    all_points   = []
    instructions = []

    def add(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    # Initial ink dip
    print("Inserting initial ink dip...")
    insert_ink_dip(all_points, instructions)

    # Approach center above skin
    center_lifted = lift_pulse(
        APPROACH_POINT.astype(int).tolist())
    instructions.append(
        f"MOVJ C{add(center_lifted):05d} "
        f"VJ={MOVEJ_SPEED:.2f}")

    elapsed_since_dip = 0.0
    total_dips        = 1
    total_draw_pts    = 0

    for contour in mapped_contours:
        if len(contour) < 2:
            continue

        first        = contour[0]
        last         = contour[-1]
        first_lifted = lift_pulse(first)
        last_lifted  = lift_pulse(last)

        if AIR_ONLY_MODE:
            contour = [lift_pulse(p) for p in contour]

        instructions.append(
            f"MOVJ C{add(first_lifted):05d} "
            f"VJ={MOVEJ_SPEED:.2f}")
        instructions.append(
            f"MOVL C{add(first):05d} "
            f"V={MOVL_SPEED:.1f}")

        for p in contour[1:]:
            elapsed_since_dip += SECONDS_PER_DRAW_POINT
            if elapsed_since_dip >= REDIP_INTERVAL_SEC:
                p_lifted = lift_pulse(p)
                instructions.append(
                    f"MOVL C{add(p_lifted):05d} "
                    f"V={MOVL_SPEED:.1f}")

                insert_ink_dip(all_points, instructions)
                total_dips       += 1
                elapsed_since_dip = 0.0

                instructions.append(
                    f"MOVJ C{add(p_lifted):05d} "
                    f"VJ={MOVEJ_SPEED:.2f}")
                instructions.append(
                    f"MOVL C{add(p):05d} "
                    f"V={MOVL_SPEED:.1f}")
            else:
                instructions.append(
                    f"MOVL C{add(p):05d} "
                    f"V={MOVL_SPEED:.1f}")

            total_draw_pts += 1

        instructions.append(
            f"MOVL C{add(last_lifted):05d} "
            f"V={MOVL_SPEED:.1f}")

    # Return to home when done
    home_idx = add(ROBOT_HOME_PULSE)
    instructions.append(
        f"MOVJ C{home_idx:05d} VJ={MOVEJ_SPEED:.2f}")
    print("  [HOME] added at end of job")

    lines = [
        "/JOB", f"//NAME {job_name}", "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0", "///POSTYPE PULSE", "///PULSE",
    ]
    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}="
            f"{p[0]},{p[1]},{p[2]},"
            f"{p[3]},{p[4]},{p[5]}")
    lines += [
        "//INST",
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW", "///GROUP1 RB1", "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8",
              newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved JBI: {filename}")
    print(f"Total points:    {len(all_points)}")
    print(f"Total draw pts:  {total_draw_pts}")
    print(f"Total ink dips:  {total_dips}")
    print(f"Redip interval:  {REDIP_INTERVAL_SEC}s")
    print(f"Tattoo size:     "
          f"{TATTOO_WIDTH_MM}mm x {TATTOO_HEIGHT_MM}mm")
    print(f"Est. draw time:  "
          f"{total_draw_pts * SECONDS_PER_DRAW_POINT:.1f}s")
    print(f"Returns to home after completion.")


# =========================================================
# MAIN
# =========================================================

def main():
    global GRID_PIXELS

    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> "
              "[points_per_contour]")
        sys.exit(1)

    image_path         = Path(sys.argv[1])
    points_per_contour = int(sys.argv[2]) \
                         if len(sys.argv) >= 3 \
                         else POINTS_PER_CONTOUR

    # ── Load calibration pixels ───────────────────────
    if os.path.exists("cal_pixels.json"):
        with open("cal_pixels.json", "r") as f:
            cal_data = json.load(f)
        GRID_PIXELS = cal_data["grid_pixels"]
        print(f"Loaded cal_pixels.json")
    else:
        print("ERROR: cal_pixels.json not found.")
        print("Run the full pipeline first to calibrate.")
        sys.exit(1)

    # ── Start camera and detect skin corners ─────────
    picam2 = start_picamera()

    skin_pixels = run_skin_corner_click(picam2)
    picam2.stop()
    cv2.destroyAllWindows()

    if not skin_pixels:
        print("Skin corner selection cancelled.")
        sys.exit(1)

    process_skin_corners(skin_pixels)
    load_skin_corners("skin_corners.json")

    # ── Load and process image ────────────────────────
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
    print(f"  {SKIN_JOB_FILE}")
    print(f"  {JOB_FILE}")
    print(f"\nSettings:")
    print(f"  Tattoo size:    "
          f"{TATTOO_WIDTH_MM}mm x {TATTOO_HEIGHT_MM}mm")
    print(f"  Lift height:    {LIFT_MM}mm")
    print(f"  AIR_ONLY_MODE:  {AIR_ONLY_MODE}")
    print(f"  Points/contour: {points_per_contour}")
    print(f"  Redip interval: {REDIP_INTERVAL_SEC}s")
    print(f"  Draw speed:     {MOVL_SPEED} mm/s")
    print(f"  Ink dip speed:  {MOVL_SPEED_INK} mm/s")


if __name__ == "__main__":
    main()