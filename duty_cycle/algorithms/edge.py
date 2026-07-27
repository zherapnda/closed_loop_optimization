import os
import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from loader import to_gray, to_color

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image2.png")

N_EDGE_SAMPLES = 20  # rows sampled along the height for each edge's line fit --
                     # noise in any single row barely moves a fit through this many points
EDGE_OFFSET = 10  # grayscale units above each row's own local baseline
EDGE_MARGIN_FRACTION = 0.25  # skip the top/bottom 25% of rows -- border/vignette artifacts live there
FINGER_L = 130
CONTRAST_WINDOW = 40  # samples on each side of a candidate x -- wide enough to span a few ridges for a stable prominence read
CONTRAST_FRACTION = 0.3  # candidate must reach this fraction of the best contrast found nearby -- matches duty_profile.py's validated cutoff


def detect_edge_points(gray, rows, from_right=False):
    points = []
    for r in rows:
        row = gray[r].astype(float)
        smoothed = np.convolve(row, np.ones(5) / 5, mode="same")
        baseline = np.median(smoothed[: min(30, len(smoothed))])
        if from_right:
            for i in range(len(smoothed) - 2, -1, -1):
                if smoothed[i] > baseline + EDGE_OFFSET and smoothed[i + 1] <= baseline + EDGE_OFFSET:
                    points.append((i, r))
                    break
        else:
            for i in range(1, len(smoothed)):
                if smoothed[i] > baseline + EDGE_OFFSET and smoothed[i - 1] <= baseline + EDGE_OFFSET:
                    points.append((i, r))
                    break
    return points


def local_contrast(signal, center, window):
    lo = max(0, center - window)
    hi = min(len(signal), center + window)
    segment = signal[lo:hi]
    peaks, _ = find_peaks(segment, distance=2, prominence=2)
    if len(peaks) == 0:
        return 0.0
    return float(np.median(peak_prominences(segment, peaks)[0]))


def detect_roi(image, roi_height=400, show=True):
    """Detect the two finger edges in image (a path to an image file, or an
    already-loaded numpy array) and return the ROI's slope and x-bounds at the
    top row (y=0), ready to hand to duty_profile.compute_duty_cycle."""

    gray = to_gray(image)

    bottom_row_index = gray.shape[0] - 1
    middle_row_index = gray.shape[0] // 2
    sample_rows = np.linspace(
        int(EDGE_MARGIN_FRACTION * bottom_row_index),
        int((1 - EDGE_MARGIN_FRACTION) * bottom_row_index),
        N_EDGE_SAMPLES,
        dtype=int,
    )

    left_points = detect_edge_points(gray, sample_rows, from_right=False)
    right_points = detect_edge_points(gray, sample_rows, from_right=True)

    print(f"Left edge: {len(left_points)}/{len(sample_rows)} rows detected")
    print(f"Right edge: {len(right_points)}/{len(sample_rows)} rows detected")

    if len(left_points) < 2 or len(right_points) < 2:
        raise RuntimeError("Not enough edge points detected.")

    rowm = gray[middle_row_index]
    smoothedm = np.convolve(rowm.astype(float), np.ones(5) / 5, mode="same")

    img = to_color(image)

    left_arr = np.array(left_points, dtype=np.int32)
    vx, vy, x0, y0 = cv2.fitLine(left_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vy < 0:  # fitLine's direction sign is arbitrary -- force "pointing down" so averaging with the other edge doesn't cancel
        vx, vy = -vx, -vy
    slope1 = vy / vx
    slope_perp = -1 / slope1
    print(f"Detected left line slope: {slope1:.4f} (rise/run)")
    print(f"Perpendicular slope: {slope_perp:.4f}")

    for (x, y) in left_points:
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

    pt1 = (int(x0 - 1000 * vx), int(y0 - 1000 * vy))
    pt2 = (int(x0 + 1000 * vx), int(y0 + 1000 * vy))
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    right_arr = np.array(right_points, dtype=np.int32)
    vx2, vy2, x02, y02 = cv2.fitLine(right_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vy2 < 0:
        vx2, vy2 = -vx2, -vy2
    slope2 = vy2 / vx2
    print(f"Detected right line slope: {slope2:.4f} (rise/run)")

    for (x, y) in right_points:
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

    pt3 = (int(x02 - 1000 * vx2), int(y02 - 1000 * vy2))
    pt4 = (int(x02 + 1000 * vx2), int(y02 + 1000 * vy2))
    cv2.line(img, pt3, pt4, (0, 255, 0), 2)

    # project each fitted line down to the bottom row for a stable anchor point,
    # instead of trusting any single raw row sample
    t1 = (bottom_row_index - y0) / vy
    edge_x1 = int(round(x0 + t1 * vx))
    t2 = (bottom_row_index - y02) / vy2
    edge_x2 = int(round(x02 + t2 * vx2))

    cv2.putText(img, f"x={edge_x1}", (max(0, edge_x1 - 20), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"x={edge_x2}", (max(0, edge_x2 - 20), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    angle_deg = 4.48
    angle_rad = np.deg2rad(angle_deg)
    horizontal_dist = float(edge_x2 - edge_x1)
    vertical_dist = np.tan(angle_rad) * horizontal_dist
    line_length = abs(int(round(horizontal_dist / np.cos(angle_rad))))

    center_y = max(0, bottom_row_index - 50)
    x_start = edge_x1
    y_start = max(0, min(img.shape[0] - 1, int(round(center_y))))
    x_end = edge_x2
    y_end = max(0, min(img.shape[0] - 1, int(round(center_y + vertical_dist))))

    cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    mid_x = (x_start + x_end) // 2
    mid_y = (y_start + y_end) // 2
    cv2.putText(
        img,
        f"{line_length}px @ {angle_deg:.2f}°",
        (max(0, mid_x - 60), max(0, mid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    avg_m = (slope1 + slope2) / 2
    avg_vy = (vy + vy2) / 2
    avg_vx = (vx + vx2) / 2
    length = 1000

    x03, y03 = edge_x1 + FINGER_L, bottom_row_index
    dx3 = int(np.cos(np.arctan(avg_m)) * length)
    dy3 = int(np.sin(np.arctan(avg_m)) * length)

    t = (middle_row_index - y03) / avg_vy
    x_mid = int(x03 + t * avg_vx)

    search_limit = min(len(smoothedm), x_mid + 300)
    candidates = list(range(x_mid, search_limit))
    contrasts = [local_contrast(smoothedm, x, CONTRAST_WINDOW) for x in candidates]
    best_contrast = max(contrasts) if contrasts else 0.0
    contrast_cutoff = CONTRAST_FRACTION * best_contrast

    x_search = x_mid
    for x, c in zip(candidates, contrasts):
        if c >= contrast_cutoff:
            x_search = x
            break

    print(f"Line at x_mid={x_mid} -> moved to x={x_search} "
          f"(contrast={local_contrast(smoothedm, x_search, CONTRAST_WINDOW):.2f}, cutoff={contrast_cutoff:.2f})")

    x03 += x_search - x_mid

    pt5 = (max(0, x03 - dx3), max(0, y03 - dy3))
    pt6 = (min(img.shape[1] - 1, x03 + dx3), min(img.shape[0] - 1, y03 + dy3))

    cv2.line(img, pt5, pt6, (128, 0, 255), 2)

    x04, y04 = edge_x2 - FINGER_L, bottom_row_index
    dx4 = int(np.cos(np.arctan(avg_m)) * length)
    dy4 = int(np.sin(np.arctan(avg_m)) * length)

    pt7 = (max(0, x04 - dx4), max(0, y04 - dy4))
    pt8 = (min(img.shape[1] - 1, x04 + dx4), min(img.shape[0] - 1, y04 + dy4))

    cv2.line(img, pt7, pt8, (128, 0, 255), 2)

    roi_top_y = 0
    roi_bottom_y = roi_height

    x03_top = x03 + (roi_top_y - y03) / avg_m
    x03_bottom = x03 + (roi_bottom_y - y03) / avg_m
    x04_top = x04 + (roi_top_y - y04) / avg_m
    x04_bottom = x04 + (roi_bottom_y - y04) / avg_m

    roi_box = np.array([
        [int(round(x03_top)), roi_top_y],
        [int(round(x04_top)), roi_top_y],
        [int(round(x04_bottom)), roi_bottom_y],
        [int(round(x03_bottom)), roi_bottom_y],
    ], dtype=np.int32)

    cv2.polylines(img, [roi_box], isClosed=True, color=(0, 255, 255), thickness=2)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "edge_point.png")
    cv2.imwrite(output_path, img)
    print(f"Saved marked image to {output_path}")

    if show:
        cv2.imshow("Detected bright pixel", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    x_left = min(x03_top, x04_top)
    x_right = max(x03_top, x04_top)

    return {
        "slope": avg_m,
        "x_left": int(round(x_left)),
        "x_right": int(round(x_right)),
    }


if __name__ == "__main__":
    detect_roi(DEFAULT_IMAGE, show=True)
