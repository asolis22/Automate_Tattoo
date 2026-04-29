import cv2
import numpy as np
import json
import os
from datetime import datetime

# ==========================================================
# 9-POINT CALIBRATION GRID
# (cart_X_mm, cart_Y_mm, cart_Z_mm, S, L, U, R, B, T)
# Grid order:
#   P1(0) P2(1) P3(2)   <- top row
#   P4(3) P5(4) P6(5)   <- middle row
#   P7(6) P8(7) P9(8)   <- bottom row
# ==========================================================

CALIBRATION_GRID = [
    (266.281, -103.412, -157.526, -19959,  17347, -46618,  4743, -21772,   6574),  # P1
    (264.545,   24.239, -155.017,   5579,  13845, -49514,   479, -21962,  -2952),  # P2
    (271.393,  154.316, -156.953,  29100,  21178, -42210, -3340, -23186, -11805),  # P3
    (376.842, -108.441, -161.854, -15287,  32940, -28399,  3300, -27100,   5268),  # P4
    (375.320,   25.230, -159.374,   4309,  30971, -30912,   590, -26561,  -2412),  # P5
    (379.385,  160.264, -158.680,  22794,  35589, -24610, -1949, -28727,  -9681),  # P6
    (488.746, -111.483, -165.885, -12411,  49128,  -7449,  2413, -34202,   4504),  # P7
    (488.185,   25.841, -165.188,   3338,  47378,  -9679,   603, -33698,  -1941),  # P8
    (490.716,  164.271, -164.305,  18692,  51863,  -3631, -1176, -35798,  -8229),  # P9
]

# The 4 grid cells — each is (TL_idx, TR_idx, BR_idx, BL_idx)
# using the index into CALIBRATION_GRID and GRID_PIXELS
GRID_CELLS = [
    (0, 1, 4, 3),  # top-left cell
    (1, 2, 5, 4),  # top-right cell
    (3, 4, 7, 6),  # bottom-left cell
    (4, 5, 8, 7),  # bottom-right cell
]

# Pixel positions of P1-P9 loaded from cal_pixels.json
GRID_PIXELS = [[0, 0]] * 9

# JBI settings
OUTPUT_JOB       = "SkinDetect.JBI"
JOB_NAME         = "SkinDetect"
ROBOT_HOME_PULSE = [-7969, 21694, -5134, 1465, -52599, 3149]
MOVEJ_SPEED      = 0.78
MOVL_SPEED       = 11.0

# Camera settings
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

SKIN_CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# ==========================================================
# BILINEAR INTERPOLATION WITHIN A GRID CELL
# ==========================================================

def bilinear_interp(u, v, val_tl, val_tr, val_br, val_bl):
    """
    Bilinear interpolation given u,v in [0,1] and 4 corner values.
    u=0 is left, u=1 is right
    v=0 is top,  v=1 is bottom
    """
    top    = (1 - u) * np.array(val_tl) + u * np.array(val_tr)
    bottom = (1 - u) * np.array(val_bl) + u * np.array(val_br)
    return (1 - v) * top + v * bottom


def find_cell_and_uv(px, py):
    """
    Find which grid cell a pixel falls in and compute
    its local u,v coordinates within that cell.
    Returns (cell_idx, u, v).
    If outside all cells, returns the nearest cell clamped.
    """
    gp = np.array(GRID_PIXELS, dtype=np.float64)

    best_cell = 0
    best_u    = 0.5
    best_v    = 0.5
    best_dist = float("inf")

    for cell_idx, (tl_i, tr_i, br_i, bl_i) in enumerate(GRID_CELLS):
        tl = gp[tl_i]
        tr = gp[tr_i]
        br = gp[br_i]
        bl = gp[bl_i]

        # Use perspective transform to find u,v within this cell
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        try:
            H   = cv2.getPerspectiveTransform(src, dst)
            pt  = np.array([[[float(px), float(py)]]],
                           dtype=np.float32)
            res = cv2.perspectiveTransform(pt, H)
            u   = float(res[0][0][0])
            v   = float(res[0][0][1])
        except Exception:
            continue

        # How far outside this cell is the point?
        u_clamped = max(0.0, min(1.0, u))
        v_clamped = max(0.0, min(1.0, v))
        dist = (u - u_clamped) ** 2 + (v - v_clamped) ** 2

        if dist < best_dist:
            best_dist = dist
            best_cell = cell_idx
            best_u    = u_clamped
            best_v    = v_clamped

    return best_cell, best_u, best_v


def pixel_to_pulse(px, py):
    """
    Convert pixel position to robot pulse using
    proper bilinear interpolation within the correct grid cell.
    """
    cell_idx, u, v = find_cell_and_uv(px, py)
    tl_i, tr_i, br_i, bl_i = GRID_CELLS[cell_idx]

    # Get pulse values at the 4 corners of this cell
    def pulse_of(i):
        p = CALIBRATION_GRID[i]
        return [p[3], p[4], p[5], p[6], p[7], p[8]]

    def cart_of(i):
        p = CALIBRATION_GRID[i]
        return [p[0], p[1], p[2]]

    pulse_tl = pulse_of(tl_i)
    pulse_tr = pulse_of(tr_i)
    pulse_br = pulse_of(br_i)
    pulse_bl = pulse_of(bl_i)

    cart_tl = cart_of(tl_i)
    cart_tr = cart_of(tr_i)
    cart_br = cart_of(br_i)
    cart_bl = cart_of(bl_i)

    # Interpolate pulse
    result_pulse = bilinear_interp(u, v,
                                   pulse_tl, pulse_tr,
                                   pulse_br, pulse_bl)

    # Interpolate cartesian for logging
    result_cart = bilinear_interp(u, v,
                                  cart_tl, cart_tr,
                                  cart_br, cart_bl)

    pulse = [int(round(x)) for x in result_pulse]
    X, Y, Z = float(result_cart[0]), float(result_cart[1]), \
               float(result_cart[2])

    return pulse, X, Y, Z

# ==========================================================
# CAMERA SETUP
# ==========================================================

def start_picamera():
    from picamera2 import Picamera2
    import time
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888",
              "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )
    picam2.configure(config)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    picam2.start()
    time.sleep(2)
    return picam2


def get_frame(picam2):
    raw = picam2.capture_array()
    return raw.copy()

# ==========================================================
# CALIBRATION MODE
# ==========================================================

clicked_cal_pts = []

def cal_mouse_callback(event, x, y, flags, param):
    global clicked_cal_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_cal_pts) < 9:
            clicked_cal_pts.append([x, y])
            print(f"  P{len(clicked_cal_pts)}: ({x}, {y})")


def run_calibration_mode(picam2):
    global clicked_cal_pts, GRID_PIXELS
    clicked_cal_pts = []

    print("\n" + "="*55)
    print("CALIBRATION MODE")
    print("="*55)
    print("\nLay calibration grid flat in workspace.")
    print("Click dots in order:")
    print("  P1=top-left  P2=top-center  P3=top-right")
    print("  P4=mid-left  P5=mid-center  P6=mid-right")
    print("  P7=bot-left  P8=bot-center  P9=bot-right")
    print("\nR=reset  ENTER=save  Q=cancel\n")

    win = "CALIBRATION"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cal_mouse_callback)

    dot_colors = [
        (0, 255, 255), (255, 255, 0), (0, 0, 255),
        (255, 0, 255), (0, 255, 0),   (255, 165, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128),
    ]

    while True:
        frame   = get_frame(picam2)
        display = frame.copy()

        if len(clicked_cal_pts) < 9:
            cv2.putText(
                display,
                f"Click P{len(clicked_cal_pts)+1}  "
                f"({len(clicked_cal_pts)}/9)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2,
            )
        else:
            cv2.putText(
                display,
                "All 9 done! Press ENTER to save.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2,
            )

        for i, pt in enumerate(clicked_cal_pts):
            color = dot_colors[i]
            cv2.circle(display, tuple(pt), 10, color, -1)
            cv2.putText(
                display, f"P{i+1}",
                (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        cv2.putText(
            display, "R=reset | ENTER=save | Q=cancel",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2,
        )

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") or key == ord("R"):
            clicked_cal_pts = []
            print("  Reset.")

        elif key == 13:
            if len(clicked_cal_pts) == 9:
                data = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"),
                    "grid_pixels": clicked_cal_pts
                }
                with open("cal_pixels.json", "w") as f:
                    json.dump(data, f, indent=4)
                GRID_PIXELS = clicked_cal_pts
                print("\nSaved cal_pixels.json")
                for i, pt in enumerate(clicked_cal_pts):
                    print(f"  P{i+1}: {pt}")
                cv2.destroyWindow(win)
                return clicked_cal_pts
            else:
                print(f"  Need 9 — only "
                      f"{len(clicked_cal_pts)} so far.")

        elif key == ord("q"):
            cv2.destroyWindow(win)
            return None

# ==========================================================
# SKIN CORNER CLICKING MODE
# ==========================================================

clicked_skin_pts = []

def skin_mouse_callback(event, x, y, flags, param):
    global clicked_skin_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_skin_pts) < 4:
            clicked_skin_pts.append([x, y])
            name = SKIN_CORNER_NAMES[len(clicked_skin_pts) - 1]
            print(f"  {name}: pixel ({x}, {y})")


def run_skin_detection(picam2):
    global clicked_skin_pts
    clicked_skin_pts = []

    print("\n" + "="*55)
    print("MARK SKIN CORNERS")
    print("="*55)
    print("\nClick the 4 corners of the fake skin:")
    print("  1: TOP-LEFT")
    print("  2: TOP-RIGHT")
    print("  3: BOTTOM-RIGHT")
    print("  4: BOTTOM-LEFT")
    print("\nR=reset  ENTER=generate JBI  Q=cancel\n")

    win = "SKIN CORNERS"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, skin_mouse_callback)

    corner_colors = [
        (0,   255, 255),
        (255, 255,   0),
        (0,     0, 255),
        (255,   0, 255),
    ]

    while True:
        frame   = get_frame(picam2)
        display = frame.copy()

        if len(clicked_skin_pts) < 4:
            name = SKIN_CORNER_NAMES[len(clicked_skin_pts)]
            cv2.putText(
                display,
                f"Click {name} corner  "
                f"({len(clicked_skin_pts)}/4)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2,
            )
        else:
            cv2.putText(
                display,
                "All 4 corners! Press ENTER to generate JBI.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 0), 2,
            )

        for i, pt in enumerate(clicked_skin_pts):
            color = corner_colors[i]
            cv2.circle(display, tuple(pt), 12, color, -1)
            cv2.putText(
                display,
                f"{SKIN_CORNER_NAMES[i]} ({pt[0]},{pt[1]})",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        if len(clicked_skin_pts) == 4:
            pts = np.array(clicked_skin_pts, dtype=np.int32)
            cv2.polylines(display, [pts], True,
                          (0, 255, 0), 2)

        cv2.putText(
            display,
            "R=reset | ENTER=generate JBI | Q=quit",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2,
        )

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") or key == ord("R"):
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

# ==========================================================
# JBI WRITER
# ==========================================================

def write_jbi(pulse_corners, filename=OUTPUT_JOB,
              job_name=JOB_NAME):
    all_points   = []
    instructions = []

    def add_pt(p):
        idx = len(all_points)
        all_points.append([int(x) for x in p])
        return idx

    h0 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(
        f"MOVJ C{h0:05d} VJ={MOVEJ_SPEED:.2f}")

    f0 = add_pt(pulse_corners[0])
    instructions.append(
        f"MOVJ C{f0:05d} VJ={MOVEJ_SPEED:.2f}")

    for pulse in pulse_corners[1:]:
        idx = add_pt(pulse)
        instructions.append(
            f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    c0 = add_pt(pulse_corners[0])
    instructions.append(
        f"MOVL C{c0:05d} V={MOVL_SPEED:.1f}")

    h1 = add_pt(ROBOT_HOME_PULSE)
    instructions.append(
        f"MOVJ C{h1:05d} VJ={MOVEJ_SPEED:.2f}")

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
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW",
        "///GROUP1 RB1",
        "NOP",
    ]
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8",
              newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"\nSaved: {filename}")

# ==========================================================
# PROCESS SKIN CORNERS
# ==========================================================

def process_skin_corners(skin_pixels):
    print("\n" + "="*55)
    print("CONVERTING PIXELS TO PULSE")
    print("="*55)

    pulse_corners = []
    results       = []

    for name, pt in zip(SKIN_CORNER_NAMES, skin_pixels):
        pulse, X, Y, Z = pixel_to_pulse(pt[0], pt[1])
        pulse_corners.append(pulse)
        results.append({
            "corner":    name,
            "pixel":     pt,
            "cartesian": [round(X, 3), round(Y, 3),
                          round(Z, 3)],
            "pulse":     pulse,
        })
        print(f"\n  {name}:")
        print(f"    pixel:     ({pt[0]}, {pt[1]})")
        print(f"    cartesian: X={X:.2f} Y={Y:.2f} Z={Z:.2f}")
        print(f"    pulse:     {pulse}")

    write_jbi(pulse_corners)

    out = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"),
        "corners": results,
    }
    with open("skin_corners.json", "w") as f:
        json.dump(out, f, indent=4)

    print(f"\nSaved: skin_corners.json")
    print(f"Done! Load {OUTPUT_JOB} onto the robot.")

# ==========================================================
# MAIN
# ==========================================================

def main():
    global GRID_PIXELS

    print("\n" + "="*55)
    print("  Skin Corner Picker  -->  SkinDetect.JBI")
    print("="*55)

    if os.path.exists("cal_pixels.json"):
        with open("cal_pixels.json", "r") as f:
            cal_data = json.load(f)
        GRID_PIXELS = cal_data["grid_pixels"]
        print(f"\nLoaded cal_pixels.json")
        for i, pt in enumerate(GRID_PIXELS):
            print(f"  P{i+1}: {pt}")
    else:
        print("\nNo cal_pixels.json — press C to calibrate first.")

    picam2 = start_picamera()

    print("\nControls:")
    print("  C     = calibration mode (click P1-P9 on grid)")
    print("  SPACE = mark skin corners and generate JBI")
    print("  Q     = quit\n")

    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()

        cv2.putText(
            preview, "Skin Corner Picker",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 255), 2,
        )

        cal_loaded = all(p != [0, 0] for p in GRID_PIXELS)
        if cal_loaded:
            for i, pt in enumerate(GRID_PIXELS):
                cv2.circle(preview, tuple(pt), 5,
                           (0, 200, 200), -1)
                cv2.putText(
                    preview, f"P{i+1}",
                    (pt[0] + 4, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 200, 200), 1,
                )
            cv2.putText(
                preview, "Calibration loaded",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 200), 1,
            )
        else:
            cv2.putText(
                preview, "No calibration -- press C first",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1,
            )

        cv2.putText(
            preview,
            "C=calibrate | SPACE=mark skin | Q=quit",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2,
        )

        cv2.imshow("MAIN", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") or key == ord("C"):
            run_calibration_mode(picam2)

        elif key == ord(" "):
            if not cal_loaded:
                print("\nCalibrate first! Press C.")
            else:
                skin_pts = run_skin_detection(picam2)
                if skin_pts:
                    process_skin_corners(skin_pts)

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()