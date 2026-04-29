import cv2
import json
import numpy as np
from datetime import datetime

# =========================================================
# FILES
# =========================================================

DETECTION_JSON = "fake_skin_detected_pulse.json"
INPUT_IMAGE = "Drawing.jpeg"
OUTPUT_JOB = "ImageOnSkin.JBI"
JOB_NAME = "ImageOnSkin"

# =========================================================
# ROBOT SETTINGS
# =========================================================

ROBOT_HOME_PULSE = [-7969, 21694, -5134, 1465, -52599, 3149]

MOVEJ_SPEED = 0.78
MOVL_SPEED = 8.0

# Drawing size inside skin
MARGIN = 0.12   # 12% margin so it does not draw on the edge

# Image processing
THRESHOLD = 120
MIN_CONTOUR_AREA = 20
POINT_SPACING = 8


# =========================================================
# LOAD SKIN DETECTION
# =========================================================

def load_skin_detection():
    with open(DETECTION_JSON, "r") as f:
        data = json.load(f)

    corners = data["robot_pulse_corners"]

    skin = {
        "TL": np.array(corners["TL"], dtype=np.float64),
        "TR": np.array(corners["TR"], dtype=np.float64),
        "BR": np.array(corners["BR"], dtype=np.float64),
        "BL": np.array(corners["BL"], dtype=np.float64),
    }

    z = data.get("specific_z_height", None)

    print("Loaded fake skin location:")
    for label, point in skin.items():
        print(f"  {label}: {point.astype(int).tolist()}")

    print(f"Specific Z from calibration: {z}")

    return skin, z


# =========================================================
# MAP IMAGE COORDINATE TO SKIN PULSE
# =========================================================

def map_uv_to_skin_pulse(u, v, skin):
    """
    u = left/right position from 0 to 1
    v = top/bottom position from 0 to 1

    This maps a drawing point into the detected fake skin area.
    """

    # Apply margin
    u = MARGIN + u * (1.0 - 2.0 * MARGIN)
    v = MARGIN + v * (1.0 - 2.0 * MARGIN)

    top = skin["TL"] + u * (skin["TR"] - skin["TL"])
    bottom = skin["BL"] + u * (skin["BR"] - skin["BL"])

    point = top + v * (bottom - top)

    return [int(round(x)) for x in point]


# =========================================================
# IMAGE TRACE
# =========================================================

def trace_image_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    # Make black drawing become white mask
    _, mask = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > MIN_CONTOUR_AREA
    ]

    h, w = img.shape[:2]

    paths = []

    for contour in contours:
        pts = contour[:, 0, :]

        sampled = []

        for i in range(0, len(pts), POINT_SPACING):
            x, y = pts[i]

            u = x / max(w - 1, 1)
            v = y / max(h - 1, 1)

            sampled.append((u, v))

        if len(sampled) > 1:
            paths.append(sampled)

    print(f"Found {len(paths)} drawing paths.")

    return paths


# =========================================================
# CONVERT IMAGE PATHS TO ROBOT PULSE PATHS
# =========================================================

def convert_paths_to_pulse(paths, skin):
    robot_paths = []

    for path in paths:
        robot_path = []

        for u, v in path:
            pulse = map_uv_to_skin_pulse(u, v, skin)
            robot_path.append(pulse)

        if len(robot_path) > 1:
            robot_paths.append(robot_path)

    return robot_paths


# =========================================================
# WRITE JBI
# =========================================================

def write_jbi(robot_paths):
    all_points = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    # Start at home
    home_idx = add_point(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{home_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    for path in robot_paths:
        if len(path) < 2:
            continue

        # Move to start of path
        start_idx = add_point(path[0])
        instructions.append(f"MOVJ C{start_idx:05d} VJ={MOVEJ_SPEED:.2f}")

        # Draw path
        for p in path[1:]:
            idx = add_point(p)
            instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    # Return home
    home_idx = add_point(ROBOT_HOME_PULSE)
    instructions.append(f"MOVJ C{home_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    lines = [
        "/JOB",
        f"//NAME {JOB_NAME}",
        "//POS",
        f"///NPOS {len(all_points)},0,0,0,0,0",
        "///TOOL 0",
        "///POSTYPE PULSE",
        "///PULSE",
    ]

    for i, p in enumerate(all_points):
        lines.append(
            f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}"
        )

    lines += [
        "//INST",
        f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        "///ATTR SC,RW",
        "///GROUP1 RB1",
        "NOP",
    ]

    lines.extend(instructions)
    lines.append("END")

    with open(OUTPUT_JOB, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI: {OUTPUT_JOB}")
    print(f"Total robot points: {len(all_points)}")


# =========================================================
# MAIN
# =========================================================

def main():
    skin, z = load_skin_detection()

    paths = trace_image_points(INPUT_IMAGE)

    robot_paths = convert_paths_to_pulse(paths, skin)

    write_jbi(robot_paths)

    print("\nDone.")
    print(f"The image was mapped inside the detected fake skin area.")
    print(f"Output job: {OUTPUT_JOB}")


if __name__ == "__main__":
    main()
