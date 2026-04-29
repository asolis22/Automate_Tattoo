import cv2
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime

# =========================================================
# FILES
# =========================================================

WORKSPACE_JSON = "fake_skin_coordinates_robot.json"

OUTPUT_CONTOUR_JSON = "mapped_contours_robot_xy.json"
OUTPUT_MAPPING_TXT  = "point_mapping_robot_xy.txt"

# =========================================================
# TUNING
# =========================================================

MIN_CONTOUR_AREA = 30
POINTS_PER_CONTOUR = 60

# =========================================================
# IMAGE CLEANUP
# =========================================================

def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def method_v1(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 90, 255, cv2.THRESH_BINARY_INV
    )

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
    ink_response = cv2.normalize(
        ink_response, None, 0, 255, cv2.NORM_MINMAX
    )

    _, binary = cv2.threshold(
        ink_response, 40, 255, cv2.THRESH_BINARY
    )

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
            img,
            0,
            pad,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

    img1 = pad_to_height(img1, h)
    img2 = pad_to_height(img2, h)

    cv2.putText(
        img1,
        "1: Thick/Fill Mode",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        img2,
        "2: Thin/Sketch Mode",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return np.hstack((img1, img2))


def choose_trace_mask(img):
    print("Select the design area, then press ENTER or SPACE.")

    roi = cv2.selectROI(
        "Select Design Area",
        img,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w > 0 and h > 0:
        cropped = img[y:y + h, x:x + w]
    else:
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
# CONTOUR TRACING
# =========================================================

def resample_closed_contour(points, target_points=60):
    points = points.astype(np.float32)

    if len(points) < 2:
        return points

    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    diffs = np.diff(points, axis=0)

    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    total = cumulative[-1]

    if total <= 0:
        return points[:1]

    targets = np.linspace(0, total, target_points, endpoint=False)

    sampled = []

    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(idx, 0), len(seg_lengths) - 1)

        start = points[idx]
        end = points[idx + 1]
        seg_len = seg_lengths[idx]

        if seg_len == 0:
            p = start
        else:
            alpha = (t - cumulative[idx]) / seg_len
            p = start + alpha * (end - start)

        sampled.append(p)

    sampled = np.array(sampled, dtype=np.float32)

    if not np.array_equal(sampled[0], sampled[-1]):
        sampled = np.vstack([sampled, sampled[0]])

    return sampled


def extract_contours(mask, min_area=30, points_per_contour=60):
    inv = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(
        inv,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise RuntimeError("No contours found.")

    kept = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            pts = contour[:, 0, :].astype(np.float32)

            pts = resample_closed_contour(
                pts,
                target_points=points_per_contour
            )

            if len(pts) >= 3:
                kept.append(pts)

    if not kept:
        raise RuntimeError(
            "Contours were found, but all were too small after filtering."
        )

    return kept


# =========================================================
# LOAD FAKE SKIN WORKSPACE JSON
# =========================================================

def load_workspace_from_json(json_path=WORKSPACE_JSON):
    with open(json_path, "r") as f:
        data = json.load(f)

    corners = data["corners_robot_xy_mm"]

    # Fake skin corners from detection JSON:
    # TL, TR, BR, BL
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
# MAP DESIGN CONTOURS TO FAKE SKIN WORKSPACE
# =========================================================

def bilinear_interp_xy(u, v, tl, tr, br, bl):
    """
    u = left to right
    v = top to bottom

    Corners:
      TL ---- TR
      |        |
      BL ---- BR
    """

    top = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br

    return (1 - v) * top + v * bottom


def all_contours_bounds(contours):
    pts = np.vstack(contours)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    return min_x, max_x, min_y, max_y


def contours_to_robot_xy(contours, tl, tr, br, bl):
    min_x, max_x, min_y, max_y = all_contours_bounds(contours)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    mapped_contours = []
    mapping = []

    global_index = 0

    for contour_index, contour in enumerate(contours):
        mapped = []

        for point_index, (x, y) in enumerate(contour):
            u = (x - min_x) / width
            v = (y - min_y) / height

            robot_xy = bilinear_interp_xy(u, v, tl, tr, br, bl)

            robot_point = {
                "x_mm": float(robot_xy[0]),
                "y_mm": float(robot_xy[1])
            }

            mapped.append(robot_point)

            mapping.append({
                "global_index": global_index,
                "contour_index": contour_index,
                "point_index": point_index,
                "image_x": float(x),
                "image_y": float(y),
                "u": float(u),
                "v": float(v),
                "robot_x_mm": float(robot_xy[0]),
                "robot_y_mm": float(robot_xy[1])
            })

            global_index += 1

        mapped_contours.append(mapped)

    return mapped_contours, mapping


# =========================================================
# DEBUG IMAGES
# =========================================================

def save_labeled_contour_image(
    mask,
    contours,
    output_path="sampled_points_labeled.png"
):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for ci, contour in enumerate(contours):
        for pi, p in enumerate(contour):
            x, y = int(round(p[0])), int(round(p[1]))

            cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)

            cv2.putText(
                debug,
                f"{ci}:{pi}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (0, 0, 255),
                1
            )

    cv2.imwrite(output_path, debug)

    print(f"Saved labeled contour image: {output_path}")


def save_travel_order_image(
    mask,
    contours,
    output_path="travel_order.png"
):
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for ci, contour in enumerate(contours):
        pts = np.array(contour, dtype=np.int32)

        for i in range(len(pts) - 1):
            cv2.line(
                debug,
                tuple(pts[i][:2]),
                tuple(pts[i + 1][:2]),
                (0, 255, 0),
                1
            )

        start = tuple(pts[0][:2])

        cv2.circle(debug, start, 7, (255, 0, 0), -1)

        cv2.putText(
            debug,
            str(ci),
            (start[0] + 4, start[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    cv2.imwrite(output_path, debug)

    print(f"Saved travel order image: {output_path}")


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_point_mapping(mapping, output_path=OUTPUT_MAPPING_TXT):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Point Mapping: Design Image -> Fake Skin Robot XY Workspace\n")
        f.write("=" * 78 + "\n\n")

        for item in mapping:
            f.write(f"Global Point {item['global_index']}\n")
            f.write(
                f"  Contour/Point : "
                f"({item['contour_index']}, {item['point_index']})\n"
            )
            f.write(
                f"  Image Point   : "
                f"({item['image_x']:.2f}, {item['image_y']:.2f})\n"
            )
            f.write(
                f"  Workspace UV  : "
                f"({item['u']:.4f}, {item['v']:.4f})\n"
            )
            f.write(
                f"  Robot XY mm   : "
                f"X={item['robot_x_mm']:.3f}, "
                f"Y={item['robot_y_mm']:.3f}\n\n"
            )

    print(f"Saved point mapping: {output_path}")


def save_robot_xy_json(
    mapped_contours,
    mapping,
    workspace_data,
    output_path=OUTPUT_CONTOUR_JSON
):
    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "Design contours mapped into fake skin workspace JSON",
        "workspace_source": WORKSPACE_JSON,
        "fake_skin_workspace": workspace_data,
        "contours_robot_xy_mm": mapped_contours,
        "flat_point_mapping": mapping
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved mapped robot XY JSON: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python trace_to_skin_workspace.py "
            "<image_path> [points_per_contour]"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])

    points_per_contour = (
        int(sys.argv[2])
        if len(sys.argv) >= 3
        else POINTS_PER_CONTOUR
    )

    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tl, tr, br, bl, workspace_data = load_workspace_from_json(WORKSPACE_JSON)

    chosen_mask = choose_trace_mask(img)

    contours = extract_contours(
        chosen_mask,
        min_area=MIN_CONTOUR_AREA,
        points_per_contour=points_per_contour
    )

    save_labeled_contour_image(chosen_mask, contours)

    save_travel_order_image(chosen_mask, contours)

    mapped_contours, mapping = contours_to_robot_xy(
        contours,
        tl,
        tr,
        br,
        bl
    )

    save_robot_xy_json(mapped_contours, mapping, workspace_data)

    save_point_mapping(mapping)

    print("\nDone.")
    print(f"Found {len(contours)} contour(s).")
    print("Saved:")
    print("  - chosen_mask.png")
    print("  - sampled_points_labeled.png")
    print("  - travel_order.png")
    print(f"  - {OUTPUT_MAPPING_TXT}")
    print(f"  - {OUTPUT_CONTOUR_JSON}")

    print("\nIMPORTANT:")
    print("This version outputs robot X/Y mm points.")
    print("It does NOT create a JBI yet.")
    print("To create a robot job, we still need fixed Z, Rx, Ry, Rz values.")


if __name__ == "__main__":
    main()
