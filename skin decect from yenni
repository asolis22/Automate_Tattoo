import cv2
import numpy as np
import os
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA = 8000

LOWER_SKIN = np.array([5, 20, 120])
UPPER_SKIN = np.array([35, 170, 255])

KERNEL_SIZE = 5

OUTPUT_IMAGE = "detected_fake_skin_result.jpg"
OUTPUT_MASK = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_results.json"
OUTPUT_POINTS_TXT = "fake_skin_results.txt"
OUTPUT_JOB = "SKINCORN.JBI"

# =========================================================
# TOOL OFFSET (CM)
# camera → needle offset
# =========================================================

TOOL_OFFSET_CM = np.array([-39.0, -16.0, -40.0])  # X-, Y-, Z-

# =========================================================
# CAMERA WORKSPACE (PIXELS)
# =========================================================

CAMERA_WORKSPACE_PIXELS = np.array([
    [100, 100],
    [900, 100],
    [900, 700],
    [100, 700],
], dtype=np.float32)

# =========================================================
# ROBOT WORKSPACE IN CM (ASSUMED CALIBRATION GRID)
# YOU MUST CALIBRATE THIS FOR REAL SYSTEM
# =========================================================

ROBOT_WORKSPACE_CM = np.array([
    [0, 0],
    [50, 0],
    [50, 50],
    [0, 50],
], dtype=np.float32)

# =========================================================
# POINT ORDERING
# =========================================================

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],      # TL
        pts[np.argmin(diff)],   # TR
        pts[np.argmax(s)],      # BR
        pts[np.argmax(diff)]    # BL
    ], dtype=np.float32)

# =========================================================
# DETECTION
# =========================================================

def detect_fake_skin(frame):
    output = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return output, mask, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return output, mask, None

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)
    box = order_points(box)

    labels = ["P1_TL", "P2_TR", "P3_BR", "P4_BL"]

    for label, pt in zip(labels, box):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(output, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(output, label, (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    detection = {
        "center": [cx, cy],
        "box": box.tolist(),
        "timestamp": datetime.now().isoformat()
    }

    return output, mask, detection

# =========================================================
# PIXEL → NORMALIZED
# =========================================================

def get_H():
    return cv2.getPerspectiveTransform(
        CAMERA_WORKSPACE_PIXELS,
        np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    )

def pixel_to_norm(pt, H):
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    r = cv2.perspectiveTransform(p, H)[0][0]
    return r[0], r[1]

# =========================================================
# NORMALIZED → CM WORKSPACE
# =========================================================

def norm_to_cm(u, v):
    top = ROBOT_WORKSPACE_CM[0] + u * (ROBOT_WORKSPACE_CM[1] - ROBOT_WORKSPACE_CM[0])
    bottom = ROBOT_WORKSPACE_CM[3] + u * (ROBOT_WORKSPACE_CM[2] - ROBOT_WORKSPACE_CM[3])
    return top + v * (bottom - top)

# =========================================================
# APPLY TOOL OFFSET
# =========================================================

def apply_tool_offset(pos_cm):
    return pos_cm + TOOL_OFFSET_CM

# =========================================================
# CM → PULSE (PLACEHOLDER SCALE)
# YOU MUST CALIBRATE THIS FOR REAL ROBOT
# =========================================================

CM_TO_PULSE = np.array([1000, 1000, 1000])  # placeholder scale

def cm_to_pulse(pos_cm):
    return (pos_cm * CM_TO_PULSE).astype(int).tolist()

# =========================================================
# PROCESS
# =========================================================

def process(frame):
    out, mask, det = detect_fake_skin(frame)

    cv2.imwrite(OUTPUT_IMAGE, out)
    cv2.imwrite(OUTPUT_MASK, mask)

    if det is None:
        print("No detection")
        return

    H = get_H()

    pulse_points = []

    for pt in det["box"]:
        u, v = pixel_to_norm(pt, H)

        cm = norm_to_cm(u, v)
        cm = apply_tool_offset(cm)

        pulse = cm_to_pulse(cm)
        pulse_points.append(pulse)

        print("CM:", cm, "PULSE:", pulse)

    with open(OUTPUT_POINTS_JSON, "w") as f:
        json.dump({
            "detection": det,
            "pulse_points": pulse_points
        }, f, indent=4)

    print("Saved output files")

# =========================================================
# MAIN
# =========================================================

def main():
    cap = cv2.VideoCapture(0)

    print("Press SPACE to capture, Q to quit")

    frame = None

    while True:
        ret, img = cap.read()
        cv2.imshow("camera", img)

        k = cv2.waitKey(1)

        if k == ord(' '):
            frame = img
            break
        if k == ord('q'):
            return

    cap.release()
    cv2.destroyAllWindows()

    process(frame)

if __name__ == "__main__":
    main()
