import cv2
import json
import numpy as np
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

COLOR_MODE       = "NO_SWAP"          # "NO_SWAP" | "RGB_TO_BGR" | "BGR_TO_RGB"
OUTPUT_JSON      = "skin_calibration.json"
ROUNDS           = 3
LABELS           = ["TL", "TR", "BR", "BL"]
CAMERA_WIDTH     = 1280
CAMERA_HEIGHT    = 720

CARTESIAN_KEYS   = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
PULSE_KEYS       = ["S", "L", "U", "R", "B", "T"]

# =========================================================
# COLOR HELPER
# =========================================================

def fix_color(frame):
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.copy()

# =========================================================
# CAMERA
# =========================================================

def start_camera():
    from picamera2 import Picamera2
    import time
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )
    picam2.configure(cfg)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    picam2.start()
    time.sleep(2)
    return picam2

def get_frame(picam2):
    return fix_color(picam2.capture_array())

# =========================================================
# MOUSE STATE
# =========================================================

clicked_pixel = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)

# =========================================================
# DRAWING HELPERS
# =========================================================

def draw_crosshair(frame, pt, color, size=18, thickness=2):
    x, y = int(pt[0]), int(pt[1])
    cv2.line(frame,  (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame,  (x, y - size), (x, y + size), color, thickness)
    cv2.circle(frame, (x, y), 5, color, -1)

def draw_confirmed(frame, confirmed_list):
    for p in confirmed_list:
        x = p["camera_pixel"]["x"]
        y = p["camera_pixel"]["y"]
        draw_crosshair(frame, (x, y), (0, 255, 0), size=20, thickness=2)
        cv2.putText(frame, p["label"], (x + 14, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

def draw_text_box(frame, lines, x, y, line_height=30, padding=10,
                  bg=(20, 20, 20), text_color=(255, 255, 255)):
    box_w = max(len(l) for l in lines) * 13 + padding * 2
    box_h = len(lines) * line_height + padding * 2
    x = min(x, CAMERA_WIDTH  - box_w - 5)
    y = min(y, CAMERA_HEIGHT - box_h - 5)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (100, 100, 100), 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x + padding, y + padding + (i + 1) * line_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)

def draw_input_overlay(frame, coord_keys, coord_values, current_field_idx, typing_buf):
    lines = ["  ENTER ROBOT COORDS  (type number + ENTER each field)"]
    lines.append("")
    for i, key in enumerate(coord_keys):
        val = coord_values.get(key, "")
        if i == current_field_idx:
            marker = ">>  "
            display_val = typing_buf + "|"
            color_hint  = "(typing)"
        elif val != "":
            marker = "    "
            display_val = val
            color_hint  = ""
        else:
            marker = "    "
            display_val = "---"
            color_hint  = ""
        lines.append(f"{marker}{key}: {display_val}  {color_hint}")

    lines.append("")
    lines.append("  BACKSPACE=delete   ENTER=confirm field")

    draw_text_box(frame, lines,
                  x=20, y=CAMERA_HEIGHT - len(lines) * 30 - 30,
                  line_height=28, padding=12,
                  bg=(20, 20, 50), text_color=(220, 220, 255))

# =========================================================
# PHASE A — click a pixel
# =========================================================

def phase_click(picam2, win, label, round_num, confirmed):
    global clicked_pixel
    clicked_pixel = None
    locked = None

    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()
        draw_confirmed(preview, confirmed)

        if clicked_pixel and locked is None:
            draw_crosshair(preview, clicked_pixel, (0, 220, 255), size=18)

        if locked:
            draw_crosshair(preview, locked, (255, 200, 0), size=22, thickness=3)

        status = "LOCKED - press ENTER to continue" if locked else \
                 ("Dot placed - SPACE=lock  or re-click" if clicked_pixel else
                  "Left-click the dot on the skin")
        draw_text_box(preview,
                      [f"Round {round_num}/{ROUNDS}   Marking: {label}",
                       status,
                       "Q = quit"],
                      x=20, y=15, bg=(30, 30, 30), text_color=(0, 255, 255))

        cv2.imshow(win, preview)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            return None
        if key == ord(" ") and clicked_pixel:
            locked        = clicked_pixel
            clicked_pixel = None
        if key == 13 and locked:
            return locked

# =========================================================
# PHASE B — type robot coordinates in OpenCV window
# =========================================================

def phase_coords(picam2, win, label, round_num, confirmed, locked_pixel, coord_type):
    keys = CARTESIAN_KEYS if coord_type == "CARTESIAN" else PULSE_KEYS

    coord_values   = {}
    current_idx    = 0
    typing_buf     = ""

    while current_idx < len(keys):
        frame   = get_frame(picam2)
        preview = frame.copy()
        draw_confirmed(preview, confirmed)
        draw_crosshair(preview, locked_pixel, (255, 200, 0), size=22, thickness=3)
        cv2.putText(preview, label,
                    (int(locked_pixel[0]) + 14, int(locked_pixel[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        draw_text_box(preview,
                      [f"Round {round_num}/{ROUNDS}   Coords for: {label}",
                       f"Mode: {coord_type}   (Q=quit)"],
                      x=20, y=15, bg=(30, 30, 30), text_color=(0, 255, 255))

        draw_input_overlay(preview, keys, coord_values, current_idx, typing_buf)

        cv2.imshow(win, preview)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            return None
        elif key == 13:
            try:
                float(typing_buf)
                coord_values[keys[current_idx]] = typing_buf
                typing_buf  = ""
                current_idx += 1
            except ValueError:
                typing_buf = ""
        elif key == 8:
            typing_buf = typing_buf[:-1]
        elif 32 <= key <= 126:
            ch = chr(key)
            if ch in "0123456789.-":
                typing_buf += ch

    return {
        "type":   coord_type,
        "values": {k: float(v) for k, v in coord_values.items()}
    }

# =========================================================
# COORD TYPE SELECTION
# =========================================================

def phase_select_coord_type(picam2, win, round_num, label):
    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()
        draw_text_box(preview,
                      [f"Round {round_num}/{ROUNDS}   Point: {label}",
                       "",
                       "  Select coordinate type:",
                       "  C = Cartesian  (X Y Z Rx Ry Rz)",
                       "  P = Pulse      (S L U R B T)",
                       "",
                       "  Q = quit"],
                      x=CAMERA_WIDTH // 2 - 200, y=CAMERA_HEIGHT // 2 - 120,
                      line_height=32, padding=16,
                      bg=(20, 40, 20), text_color=(100, 255, 100))
        cv2.imshow(win, preview)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            return None
        if key == ord("c"):
            return "CARTESIAN"
        if key == ord("p"):
            return "PULSE"

# =========================================================
# BETWEEN-ROUND SCREEN
# =========================================================

def phase_next_round(picam2, win, round_num):
    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()
        draw_text_box(preview,
                      [f"Round {round_num - 1} complete!",
                       "",
                       f"Move the skin to a NEW position",
                       f"for Round {round_num} of {ROUNDS}.",
                       "",
                       "Press SPACE when ready   |   Q = quit"],
                      x=CAMERA_WIDTH // 2 - 230, y=CAMERA_HEIGHT // 2 - 100,
                      line_height=34, padding=18,
                      bg=(40, 20, 20), text_color=(255, 180, 80))
        cv2.imshow(win, preview)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            return True

# =========================================================
# STARTUP SCREEN
# =========================================================

def phase_start(picam2, win):
    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()
        draw_text_box(preview,
                      ["  FAKE-SKIN  CALIBRATION",
                       "",
                       f"  {ROUNDS} rounds  x  {len(LABELS)} points each",
                       "  Point order: " + "  ".join(LABELS),
                       "",
                       "  For each point:",
                       "  1. Left-click the dot on the skin",
                       "  2. SPACE to lock the crosshair",
                       "  3. ENTER to go to coord entry",
                       "  4. Type each coord + ENTER",
                       "",
                       "  Press SPACE to begin   |   Q = quit"],
                      x=CAMERA_WIDTH // 2 - 240, y=50,
                      line_height=30, padding=18,
                      bg=(10, 10, 40), text_color=(200, 220, 255))
        cv2.imshow(win, preview)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            return True

# =========================================================
# HOMOGRAPHY
# =========================================================

def compute_homographies(rounds_data):
    results = []
    for rnd in rounds_data:
        pts_cam, pts_robot = [], []
        ok = True
        for lbl in LABELS:
            sp = rnd["skin_points"][lbl]
            px = sp["camera_pixel"]
            rp = sp["robot_position"]
            if rp["type"] == "CARTESIAN":
                rx, ry = rp["values"]["X"], rp["values"]["Y"]
            elif rp["type"] == "PULSE":
                rx, ry = rp["values"]["S"], rp["values"]["L"]
            else:
                ok = False; break
            pts_cam.append([px["x"], px["y"]])
            pts_robot.append([rx, ry])

        if ok and len(pts_cam) == 4:
            H, _ = cv2.findHomography(
                np.array(pts_cam,   dtype=np.float32),
                np.array(pts_robot, dtype=np.float32)
            )
            results.append(H.tolist() if H is not None else None)
        else:
            results.append(None)
    return results

# =========================================================
# SAVE
# =========================================================

def save_json(data):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nCalibration saved -> {OUTPUT_JSON}")

# =========================================================
# MAIN
# =========================================================

def main():
    global clicked_pixel

    calibration_data = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "description":      "Fake-skin camera-to-robot calibration",
        "camera_width":     CAMERA_WIDTH,
        "camera_height":    CAMERA_HEIGHT,
        "color_mode":       COLOR_MODE,
        "round_count":      ROUNDS,
        "point_order":      LABELS,
        "rounds":           [],
        "homographies":     []
    }

    picam2 = start_camera()
    WIN    = "Skin Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_callback)

    try:
        if not phase_start(picam2, WIN):
            return

        for round_num in range(1, ROUNDS + 1):

            if round_num > 1:
                if not phase_next_round(picam2, WIN, round_num):
                    return

            confirmed = []

            for label in LABELS:

                # 1. pick coord type
                coord_type = phase_select_coord_type(picam2, WIN, round_num, label)
                if coord_type is None:
                    return

                # 2. click & lock pixel
                clicked_pixel = None
                locked = phase_click(picam2, WIN, label, round_num, confirmed)
                if locked is None:
                    return

                # 3. type coords in window
                robot_pos = phase_coords(picam2, WIN, label, round_num,
                                         confirmed, locked, coord_type)
                if robot_pos is None:
                    return

                confirmed.append({
                    "label":          label,
                    "camera_pixel":   {"x": int(locked[0]), "y": int(locked[1])},
                    "robot_position": robot_pos
                })

            calibration_data["rounds"].append({
                "round_number": round_num,
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
                "skin_points":  {p["label"]: {
                                    "camera_pixel":   p["camera_pixel"],
                                    "robot_position": p["robot_position"]
                                 } for p in confirmed},
                "points": confirmed
            })

        calibration_data["homographies"] = compute_homographies(
            calibration_data["rounds"]
        )
        save_json(calibration_data)

        # completion screen
        while True:
            frame   = get_frame(picam2)
            preview = frame.copy()
            draw_text_box(preview,
                          ["  CALIBRATION COMPLETE!",
                           "",
                           f"  Saved: {OUTPUT_JSON}",
                           f"  {ROUNDS} rounds  x  {len(LABELS)} points",
                           "",
                           "  Press Q to exit"],
                          x=CAMERA_WIDTH // 2 - 200, y=CAMERA_HEIGHT // 2 - 100,
                          line_height=34, padding=18,
                          bg=(10, 40, 10), text_color=(80, 255, 80))
            cv2.imshow(WIN, preview)
            if cv2.waitKey(16) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()