import cv2
import numpy as np
import os
import json
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

MIN_AREA = 50000  # false positives ~15k px; beige skin is ~145k px — big gap

# HSV range measured from actual beige/tan fake skin screenshots.
# Skin: H~25 (warm beige), S~84 (low-mid saturation), V~232 (bright).
# Foam board: H~87, S~101 — different enough in HUE to separate cleanly.
# Using H[10-35] keeps us in warm beige territory, well away from foam (H~87).
# S minimum kept low (40) because beige is not very saturated.
# V minimum 150 to exclude shadows and dark pen marks on the skin.
LOWER_SKIN = np.array([10, 40, 150])
UPPER_SKIN = np.array([35, 255, 255])

KERNEL_SIZE = 5

OUTPUT_IMAGE       = "detected_fake_skin_result.jpg"
OUTPUT_MASK        = "detected_fake_skin_mask.jpg"
OUTPUT_POINTS_JSON = "fake_skin_corners_and_bbox.json"
OUTPUT_POINTS_TXT  = "fake_skin_corners_and_bbox.txt"
OUTPUT_JOB         = "SKINCORN.JBI"
JOB_NAME           = "SKINCORN"

CAMERA_WORKSPACE_FILE = "camera_workspace_pixels.json"

COLOR_MODE = "NO_SWAP"  # Raw feed - no channel swap on preview

# =========================================================
# ROBOT CALIBRATION POINTS
# Order: TL, TR, BR, BL  (6 pulse values each)
# =========================================================

# Calibration points from TATCALIBRATE.JBI (2026/04/27)
# C00001=TL, C00002=TR, C00003=BR, C00004=BL
# C00000 is the safe home/approach point (not used for workspace mapping)
ROBOT_WORKSPACE_PULSE = [
    [-43487, 10808, -30891, 5060, -37212, 17750],   # TL  (C00001)
    [-24167, 57720,  22702, 2808, -52785, 10258],   # TR  (C00002)
    [ 23516, 61116,  27250,-1344, -54873,-10642],   # BR  (C00003)
    [ 42600, 11752, -30041,-3542, -38135,-18280],   # BL  (C00004)
]

# =========================================================
# CAMERA → NEEDLE PHYSICAL OFFSET
# =========================================================
# The tattoo needle is physically offset from the camera center:
#   16.5 cm to the LEFT  of the camera
#   38.5 cm DOWN         from the camera (height/Z difference)
#   39.5 cm              from needle tip to skin surface (approach depth)
#
# We apply the LEFT offset in normalized UV space (0.0-1.0).
# To do this we need the workspace width in cm.
# TL pulse S-axis = -43487, BL pulse S-axis = 42600
# S-axis range = 86087 pulses across the workspace.
# We store the raw cm offset and convert to UV fraction at runtime
# using the actual pulse range of the workspace.
#
# Horizontal (left/right) offset: needle is 16.5cm LEFT of camera
# We compute UV shift = 16.5 / workspace_width_cm
# For now we store it and apply directly in pulse interpolation.
#
# The 38.5cm vertical and 39.5cm depth are Z/height offsets —
# these affect how high the robot needs to be, not XY position.
# They are noted here for reference when tuning approach height.

CAM_TO_NEEDLE_LEFT_CM  = 16.5   # needle is this far LEFT of camera
CAM_TO_NEEDLE_DOWN_CM  = 38.5   # needle is this far below camera (Z)
NEEDLE_TO_SKIN_CM      = 39.5   # needle tip to skin surface distance

# =========================================================
# JBI TUNING  (match the working TimYas values)
# =========================================================

MOVEJ_SPEED = 0.78   # VJ for repositioning jumps
MOVL_SPEED  = 20.0   # V  for linear moves (confirmed from PENDOWN.JBI)
LIFT_J3     = 1400   # pulse counts to lift J3 for pen-up (if needed)

# =========================================================
# COLOR FIX
# =========================================================

def fix_pi_camera_color(frame):
    """
    The Pi camera captures in RGB888. Picamera2 gives us RGB, but OpenCV
    expects BGR. So RGB_TO_BGR is almost always correct for Picamera2.
    If the preview looks orange/wrong, try key 1 (RGB_TO_BGR).
    """
    if COLOR_MODE == "RGB_TO_BGR":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif COLOR_MODE == "BGR_TO_RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return frame.copy()


def improve_color_for_detection(frame_bgr):
    """
    Prepares frame FOR DETECTION ONLY - never shown to user.
    The blue board dominates the scene, so we boost saturation
    to help the yellow fake skin pop in HSV space.
    No white balance - that was causing the crazy color inversions.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.40, 0, 255)  # boost saturation for mask
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # slight brightness
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# =========================================================
# CAMERA
# =========================================================

def start_picamera():
    from picamera2 import Picamera2
    import time

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)

    # Camera Module 2 has FIXED FOCUS — no AfMode support.
    # We only set auto white balance and auto exposure.
    # Focus is set physically by rotating the lens on the camera module.
    picam2.set_controls({
        "AwbEnable": True,
        "AeEnable":  True,
    })
    picam2.start()

    # Give AE/AWB time to settle before first frame
    time.sleep(2)
    return picam2


def get_frame(picam2):
    """Return the raw frame with NO color modification for the preview.
    Color correction is only applied internally during HSV detection."""
    raw = picam2.capture_array()
    return fix_pi_camera_color(raw)


# =========================================================
# POINT ORDERING
# =========================================================

def order_points(pts):
    pts  = np.array(pts, dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl   = pts[np.argmin(s)]
    tr   = pts[np.argmin(diff)]
    br   = pts[np.argmax(s)]
    bl   = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# =========================================================
# CLICK CAMERA WORKSPACE CALIBRATION
# =========================================================

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: [{x}, {y}]")


def calibrate_camera_workspace(picam2):
    global clicked_points
    clicked_points = []

    print("\nCAMERA WORKSPACE CALIBRATION")
    print("Click the FULL robot workspace corners in order:")
    print("  1. Top Left\n  2. Top Right\n  3. Bottom Right\n  4. Bottom Left")
    print("Press Q to cancel.\n")

    cv2.namedWindow("Click Workspace Corners: TL, TR, BR, BL")
    cv2.setMouseCallback("Click Workspace Corners: TL, TR, BR, BL", mouse_callback)

    labels = ["TL", "TR", "BR", "BL"]
    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()

        for i, pt in enumerate(clicked_points):
            x, y = pt
            cv2.circle(preview, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(preview, labels[i], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(preview, f"Clicked {len(clicked_points)}/4 corners",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Click Workspace Corners: TL, TR, BR, BL", preview)

        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            break
        if key == ord("q"):
            cv2.destroyWindow("Click Workspace Corners: TL, TR, BR, BL")
            return None

    cv2.destroyWindow("Click Workspace Corners: TL, TR, BR, BL")

    workspace_pixels = np.array(clicked_points, dtype=np.float32)
    with open(CAMERA_WORKSPACE_FILE, "w") as f:
        json.dump({
            "order": ["TL", "TR", "BR", "BL"],
            "camera_workspace_pixels": workspace_pixels.astype(int).tolist()
        }, f, indent=4)

    print(f"\nSaved camera workspace calibration to {CAMERA_WORKSPACE_FILE}")
    print(workspace_pixels)
    return workspace_pixels


# Camera workspace = full camera frame edges (1280x720)
# The 4 robot pulse points correspond to the 4 edges of what the camera sees:
# TL=(0,0), TR=(1280,0), BR=(1280,720), BL=(0,720)
CAMERA_WORKSPACE_PIXELS = np.array([
    [0,    0  ],   # TL — corresponds to ROBOT_WORKSPACE_PULSE[0]
    [1280, 0  ],   # TR — corresponds to ROBOT_WORKSPACE_PULSE[1]
    [1280, 720],   # BR — corresponds to ROBOT_WORKSPACE_PULSE[2]
    [0,    720],   # BL — corresponds to ROBOT_WORKSPACE_PULSE[3]
], dtype=np.float32)

def load_camera_workspace_pixels():
    return CAMERA_WORKSPACE_PIXELS


# =========================================================
# FAKE SKIN DETECTION
# =========================================================

def detect_fake_skin_rotated_box(frame):
    output = frame.copy()

    # Apply color correction ONLY for detection - never touch the preview image
    detection_frame = improve_color_for_detection(frame)
    hsv  = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return output, mask, None

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    if area < MIN_AREA:
        return output, mask, None

    rect           = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    box         = cv2.boxPoints(rect)
    box         = np.int32(box)
    ordered_box = order_points(box)

    cv2.drawContours(output, [largest],              -1, (255, 0, 0), 2)
    cv2.drawContours(output, [ordered_box.astype(int)], 0, (0, 255, 0), 3)

    labels = ["P1_TL", "P2_TR", "P3_BR", "P4_BL"]
    for label, pt in zip(labels, ordered_box):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(output, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(output, f"{label}: ({x},{y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    p_points = {
        "P1_TL": [int(ordered_box[0][0]), int(ordered_box[0][1])],
        "P2_TR": [int(ordered_box[1][0]), int(ordered_box[1][1])],
        "P3_BR": [int(ordered_box[2][0]), int(ordered_box[2][1])],
        "P4_BL": [int(ordered_box[3][0]), int(ordered_box[3][1])],
    }

    detection_info = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "center_px":    [float(cx), float(cy)],
        "bounding_box": {
            "width_px":  float(w),
            "height_px": float(h),
            "angle_deg": float(angle),
            "area_px":   float(area),
        },
        "corner_points_px":       p_points,
        "box_points_ordered_list": ordered_box.tolist(),
    }

    return output, mask, detection_info


# =========================================================
# CAMERA PIXEL → ROBOT PULSE
# =========================================================

def get_camera_to_normalized_workspace_matrix(camera_workspace_pixels):
    normalized_workspace = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(camera_workspace_pixels, normalized_workspace)


def pixel_to_normalized(pixel_point, H):
    src = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0][0][0]), float(dst[0][0][1])


def bilinear_pulse_from_normalized(u, v):
    """
    Bilinear interpolation across the four calibrated robot corners.
      u = 0 → left edge,  u = 1 → right edge
      v = 0 → top edge,   v = 1 → bottom edge

    Applies camera-to-needle offset in pulse space AFTER interpolation.
    The needle is 16.5cm to the LEFT of the camera, so we compute
    what that left shift equals in pulse units by interpolating the
    difference between TL and TR pulse vectors.
    """
    if u < -0.20 or u > 1.20 or v < -0.20 or v > 1.20:
        raise ValueError(
            f"Point too far outside calibrated workspace: u={u:.3f}, v={v:.3f}"
        )

    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    TL = np.array(ROBOT_WORKSPACE_PULSE[0], dtype=np.float64)
    TR = np.array(ROBOT_WORKSPACE_PULSE[1], dtype=np.float64)
    BR = np.array(ROBOT_WORKSPACE_PULSE[2], dtype=np.float64)
    BL = np.array(ROBOT_WORKSPACE_PULSE[3], dtype=np.float64)

    top    = TL + u * (TR - TL)
    bottom = BL + u * (BR - BL)
    point  = top + v * (bottom - top)

    # ── apply camera-to-needle LEFT offset in pulse space ──────────────────
    # The workspace spans from TL to TR horizontally.
    # We compute the pulse vector for "1 full workspace width" on the top edge,
    # then scale it by the fraction (16.5cm / workspace_width_cm).
    # Since we don't know workspace_width_cm, we express the offset as a
    # fraction of the workspace using the S-axis (joint 1) pulse range.
    # S-axis: TL[0]=-43487, TR[0]=-24167  → width = 19320 pulses on top edge
    #         BL[0]= 42600, BR[0]= 23516  → width = 19084 pulses on bottom edge
    # We'll use the average: ~19200 pulses = full workspace width.
    # The offset vector (left = toward TL from TR) at current v:
    left_vec_top    = TL - TR          # pulse vector pointing LEFT on top edge
    left_vec_bottom = BL - BR          # pulse vector pointing LEFT on bottom edge
    left_vec        = left_vec_top + v * (left_vec_bottom - left_vec_top)

    # Normalize to per-pulse-width, then scale by cm offset fraction.
    # We use S-axis width as the reference (index 0 = S axis).
    workspace_s_width = abs(TR[0] - TL[0]) + v * (abs(BR[0] - BL[0]) - abs(TR[0] - TL[0]))
    if workspace_s_width > 0:
        # offset_fraction = how far LEFT as fraction of workspace width
        # We don't know physical cm width, so we note:
        # CAM_TO_NEEDLE_LEFT_CM = 16.5cm stored for when physical width is measured.
        # For now apply as a fixed pulse offset derived from the S-axis geometry.
        offset_fraction = CAM_TO_NEEDLE_LEFT_CM / 60.0  # assume ~60cm workspace width
        point = point + offset_fraction * left_vec

    return [int(round(x)) for x in point]


# =========================================================
# LIFT HELPER  (matches TimYas approach)
# =========================================================

def make_lifted_pulse(draw_pulse):
    """Return a copy of draw_pulse with J3 raised by LIFT_J3."""
    lifted    = list(draw_pulse)
    lifted[2] += LIFT_J3
    return lifted


# =========================================================
# JBI WRITER  — fixed format matching the working TimYas file
# =========================================================

def write_jbi_corner_visit(robot_pulse_points, filename=OUTPUT_JOB, job_name=JOB_NAME):
    """
    Generates a Yaskawa JBI that visits all four skin corners in order:
      TL → TR → BR → BL → back to TL
    Approach / depart via a lifted version of TL (safe travel height).

    Point-list is built dynamically so NPOS always matches reality.
    All positions are PULSE type with 6 axes.
    """

    if len(robot_pulse_points) != 4:
        raise ValueError(f"Expected 4 corner pulse points, got {len(robot_pulse_points)}")

    for i, p in enumerate(robot_pulse_points):
        if len(p) != 6:
            raise ValueError(f"Point {i} has {len(p)} values — must be 6 (S,L,U,R,B,T).")

    # ── build point list & instructions dynamically ──────────────────────────
    all_points   = []
    instructions = []

    def add_point(p):
        idx = len(all_points)
        all_points.append([int(v) for v in p])
        return idx

    # MOVJ to TL first (safe repositioning joint move, matches PENDOWN pattern)
    tl_idx = add_point(robot_pulse_points[0])
    instructions.append(f"MOVJ C{tl_idx:05d} VJ={MOVEJ_SPEED:.2f}")

    # MOVL across TR → BR → BL (linear precision moves)
    for pulse in robot_pulse_points[1:]:
        idx = add_point(pulse)
        instructions.append(f"MOVL C{idx:05d} V={MOVL_SPEED:.1f}")

    # Close the rectangle back to TL
    close_idx = add_point(robot_pulse_points[0])
    instructions.append(f"MOVL C{close_idx:05d} V={MOVL_SPEED:.1f}")

    # ── assemble JBI text ─────────────────────────────────────────────────────
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
    lines.append(f"///DATE {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.extend(instructions)
    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}  ({len(all_points)} position(s), "
          f"{len(instructions)} instruction(s))")


# =========================================================
# SAVE HELPER FILES
# =========================================================

def save_detection_files(detection_info, robot_pulse_points):
    detection_info["robot_pulse_points"] = {
        "P1_TL": robot_pulse_points[0],
        "P2_TR": robot_pulse_points[1],
        "P3_BR": robot_pulse_points[2],
        "P4_BL": robot_pulse_points[3],
    }

    with open(OUTPUT_POINTS_JSON, "w") as f:
        json.dump(detection_info, f, indent=4)

    with open(OUTPUT_POINTS_TXT, "w") as f:
        f.write("FAKE SKIN DETECTION RESULTS\n")
        f.write("===========================\n\n")
        f.write(f"Timestamp: {detection_info['timestamp']}\n\n")

        f.write("Bounding Box:\n")
        f.write(f"  Width px:  {detection_info['bounding_box']['width_px']:.2f}\n")
        f.write(f"  Height px: {detection_info['bounding_box']['height_px']:.2f}\n")
        f.write(f"  Angle deg: {detection_info['bounding_box']['angle_deg']:.2f}\n")
        f.write(f"  Area px:   {detection_info['bounding_box']['area_px']:.2f}\n\n")

        f.write("P1-P4 Camera Pixel Corners:\n")
        for name, pt in detection_info["corner_points_px"].items():
            f.write(f"  {name}: {pt}\n")

        f.write("\nP1-P4 Yaskawa Pulse Points:\n")
        for name, pulse in detection_info["robot_pulse_points"].items():
            f.write(f"  {name}: {pulse}\n")

    print(f"Saved {OUTPUT_POINTS_JSON}")
    print(f"Saved {OUTPUT_POINTS_TXT}")


# =========================================================
# MAIN PROCESS
# =========================================================

def process_frame(frame, camera_workspace_pixels):
    annotated, mask, info = detect_fake_skin_rotated_box(frame)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(OUTPUT_MASK, mask)

    if info is None:
        print("\nNo fake skin detected.")
        print(f"Saved {OUTPUT_IMAGE} and {OUTPUT_MASK}")
        return

    print("\nDetected fake skin.")
    print(f"\nBounding box:")
    print(f"  Width px:  {info['bounding_box']['width_px']:.2f}")
    print(f"  Height px: {info['bounding_box']['height_px']:.2f}")
    print(f"  Angle deg: {info['bounding_box']['angle_deg']:.2f}")
    print(f"  Area px:   {info['bounding_box']['area_px']:.2f}")

    print("\nP1-P4 camera pixel corners:")
    for label, pt in info["corner_points_px"].items():
        print(f"  {label}: {pt}")

    H = get_camera_to_normalized_workspace_matrix(camera_workspace_pixels)

    robot_pulse_points = []
    print("\nConverted fake skin corners to Yaskawa pulse positions:")

    for label, pixel_pt in info["corner_points_px"].items():
        u, v   = pixel_to_normalized(pixel_pt, H)
        print(f"  {label} normalized: u={u:.4f}, v={v:.4f}")
        pulse  = bilinear_pulse_from_normalized(u, v)
        robot_pulse_points.append(pulse)
        print(f"  {label} pulse: {pulse}")

    write_jbi_corner_visit(robot_pulse_points, filename=OUTPUT_JOB, job_name=JOB_NAME)
    save_detection_files(info, robot_pulse_points)

    print("\nSaved files:")
    print(f"  {OUTPUT_IMAGE}")
    print(f"  {OUTPUT_MASK}")
    print(f"  {OUTPUT_POINTS_JSON}")
    print(f"  {OUTPUT_POINTS_TXT}")
    print(f"  {OUTPUT_JOB}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n=== Fake Skin Detection → Yaskawa Corner JBI ===\n")

    picam2 = start_picamera()

    camera_workspace_pixels = load_camera_workspace_pixels()

    print("Controls:")
    print("  C     = calibrate / click workspace corners (TL TR BR BL)")
    print("  SPACE = capture frame, detect skin, generate JBI")
    print("  Q     = quit")
    print("  1     = RGB_TO_BGR color mode")
    print("  2     = NO_SWAP  color mode  (default)")
    print("  3     = BGR_TO_RGB color mode")
    print("  NOTE  = Camera Module 2 has FIXED FOCUS.")
    print("          If image is blurry at your distance, physically rotate")
    print("          the small lens on the camera module to adjust focus.\n")

    print("Camera workspace = full image frame (hardcoded).")
    print("Robot pulse corners correspond to camera view edges.\n")

    global COLOR_MODE

    while True:
        frame   = get_frame(picam2)
        preview = frame.copy()

        cv2.putText(preview, f"COLOR_MODE: {COLOR_MODE}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if camera_workspace_pixels is not None:
            pts = camera_workspace_pixels.astype(int)
            cv2.polylines(preview, [pts], True, (255, 0, 255), 3)
            for label, pt in zip(["TL", "TR", "BR", "BL"], pts):
                cv2.circle(preview, tuple(pt), 8, (255, 0, 255), -1)
                cv2.putText(preview, label, (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Pi Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            COLOR_MODE = "RGB_TO_BGR"
            print("COLOR_MODE = RGB_TO_BGR")

        elif key == ord("2"):
            COLOR_MODE = "NO_SWAP"  # Raw feed - no channel swap on preview
            print("COLOR_MODE = NO_SWAP")

        elif key == ord("3"):
            COLOR_MODE = "BGR_TO_RGB"
            print("COLOR_MODE = BGR_TO_RGB")

        elif key == ord("c"):
            camera_workspace_pixels = calibrate_camera_workspace(picam2)

        elif key == ord(" "):
            process_frame(frame.copy(), CAMERA_WORKSPACE_PIXELS)

        elif key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()