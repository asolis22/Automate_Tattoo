import cv2
import numpy as np
import sys
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================

BOX_WIDTH_IN = 5.0
BOX_HEIGHT_IN = 6.0
MM_PER_IN = 25.4

BOX_WIDTH_MM = BOX_WIDTH_IN * MM_PER_IN     # 127.0 mm
BOX_HEIGHT_MM = BOX_HEIGHT_IN * MM_PER_IN   # 152.4 mm

DEFAULT_POINTS = 18
JOB_NAME = "XYTRACE"
JOB_FILE = "XYTRACE.JBI"

USER_FRAME_NO = 1
TOOL_NO = 0

# Keep this above the surface if your UF origin is on the table/sheet.
Z_HEIGHT_MM = 30.0

# Orientation: keep constant for now
RX = 180.0
RY = 0.0
RZ = 0.0

# Top-left origin inside the box in the UF plane
X_OFFSET_MM = 0.0
Y_OFFSET_MM = 0.0


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
    if w > 0 and h > 0:
        cropped = img[y:y+h, x:x+w]
    else:
        print("No ROI selected. Using full image.")
        cropped = img.copy()

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
# SAMPLE POINTS BY DISTANCE
# =========================================================

def sample_contour_points_by_distance(mask, num_points=18):
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contour found.")

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :].astype(np.float32)

    if len(contour) < 2:
        raise RuntimeError("Contour too small.")

    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        raise RuntimeError("Contour length is zero.")

    targets = np.linspace(0, total_length, num_points, endpoint=False)

    sampled = []
    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(idx, 0), len(seg_lengths) - 1)

        seg_start = contour[idx]
        seg_end = contour[idx + 1]
        seg_len = seg_lengths[idx]

        if seg_len == 0:
            p = seg_start
        else:
            local_t = (t - cumulative[idx]) / seg_len
            p = seg_start + local_t * (seg_end - seg_start)

        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)
    sampled = np.vstack([sampled, sampled[0]])  # close loop
    return sampled.astype(np.float32)


def save_labeled_point_image(mask, sampled_points, output_path="sampled_points_labeled.png"):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, p in enumerate(sampled_points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(debug, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(debug, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imwrite(output_path, debug)
    print(f"Saved labeled point image: {output_path}")


# =========================================================
# IMAGE POINTS -> XY USER FRAME POINTS
# =========================================================

def image_points_to_xy_mm(sampled_points,
                          box_width_mm=127.0,
                          box_height_mm=152.4,
                          x_offset_mm=0.0,
                          y_offset_mm=0.0,
                          z_height_mm=30.0):
    """
    Scale sampled image points to fit inside a box in the USER FRAME XY plane.
    Keeps aspect ratio.
    """
    pts = sampled_points.astype(np.float32)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    src_w = max(max_x - min_x, 1.0)
    src_h = max(max_y - min_y, 1.0)

    scale = min(box_width_mm / src_w, box_height_mm / src_h)

    scaled_w = src_w * scale
    scaled_h = src_h * scale

    # center inside the 5x6 box
    x_margin = (box_width_mm - scaled_w) / 2.0
    y_margin = (box_height_mm - scaled_h) / 2.0

    xy_points = []
    mapping = []

    for i, (x, y) in enumerate(pts):
        nx = (x - min_x) * scale
        ny = (y - min_y) * scale

        # image y goes downward, so flip if you want a more natural XY plane
        x_mm = x_offset_mm + x_margin + nx
        y_mm = y_offset_mm + (box_height_mm - (y_margin + ny))

        point = [x_mm, y_mm, z_height_mm, RX, RY, RZ]
        xy_points.append(point)

        mapping.append({
            "index": i,
            "image_x": float(x),
            "image_y": float(y),
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_height_mm,
            "rx": RX,
            "ry": RY,
            "rz": RZ,
        })

    return xy_points, mapping, scale


def save_point_mapping(mapping, output_path="point_mapping_xy.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping (Image -> USER Frame XY mm)\n")
        f.write("=" * 72 + "\n\n")

        for item in mapping:
            f.write(f"Point {item['index']}\n")
            f.write(f"  Image Point : ({item['image_x']:.2f}, {item['image_y']:.2f})\n")
            f.write(
                f"  XY Point    : ({item['x_mm']:.3f}, {item['y_mm']:.3f}, {item['z_mm']:.3f}, "
                f"{item['rx']:.4f}, {item['ry']:.4f}, {item['rz']:.4f})\n\n"
            )

    print(f"Saved point mapping: {output_path}")


# =========================================================
# JBI WRITER - USER / RECTAN
# =========================================================

def write_jbi_user_job(points, filename=JOB_FILE, job_name=JOB_NAME,
                       user_frame_no=1, tool_no=0):
    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS 0,0,0,{len(points)},0,0")
    lines.append(f"///USER {user_frame_no}")
    lines.append(f"///TOOL {tool_no}")
    lines.append("///POSTYPE USER")
    lines.append("///RECTAN")
    lines.append("///RCONF 0,0,0,0,0,0,0,0")

    for i, p in enumerate(points):
        lines.append(
            f"P{i:03d}={p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{p[3]:.4f},{p[4]:.4f},{p[5]:.4f}"
        )

    lines.append("//INST")
    lines.append("///DATE 2026/04/08 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.append("MOVJ P000 VJ=5.00")

    for i in range(1, len(points)):
        lines.append(f"MOVL P{i:03d} V=60.0")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python curve_trace_to_yaskawa_xy.py <image_path> [num_points]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_POINTS

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    chosen_mask = choose_trace_mask(img)
    sampled_points = sample_contour_points_by_distance(chosen_mask, num_points=num_points)

    save_labeled_point_image(chosen_mask, sampled_points)

    xy_points, mapping, scale = image_points_to_xy_mm(
        sampled_points,
        box_width_mm=BOX_WIDTH_MM,
        box_height_mm=BOX_HEIGHT_MM,
        x_offset_mm=X_OFFSET_MM,
        y_offset_mm=Y_OFFSET_MM,
        z_height_mm=Z_HEIGHT_MM,
    )

    save_point_mapping(mapping)
    write_jbi_user_job(
        xy_points,
        filename=JOB_FILE,
        job_name=JOB_NAME,
        user_frame_no=USER_FRAME_NO,
        tool_no=TOOL_NO,
    )

    print("\nDone.")
    print(f"Scale used: {scale:.4f} mm/pixel")
    print(f"Bounding box target: {BOX_WIDTH_MM:.1f} mm x {BOX_HEIGHT_MM:.1f} mm")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - point_mapping_xy.txt")
    print(f"  - {JOB_FILE}")


if __name__ == "__main__":
    main()