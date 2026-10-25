import os
import cv2
import numpy as np
from loader import to_gray, to_color

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image2.png")

N_EDGE_SAMPLES = 20  # rows sampled along the height for each edge's line fit --
                     # noise in any single row barely moves a fit through this many points
EDGE_OFFSET = 10  # grayscale units above each row's own local baseline
EDGE_MARGIN_FRACTION = 0.25  # skip the top/bottom 25% of rows -- border/vignette artifacts live there

# physical calibration of the ROI, not a fitted/heuristic value. edge_x1/edge_x2
# (the two fitted line-of-best-fit edges) are the two FURTHEST edges of the
# metal fingers -- a known physical gap of TOTAL_FINGER_GAP_UM. The ROI we
# actually want is the gap between the TIPS of two opposite fingers, a known
# physical gap of FINGER_TIP_GAP_UM, centered in the middle of the full span.
TOTAL_FINGER_GAP_UM = 90
FINGER_TIP_GAP_UM = 45
FINGER_L_FRACTION = (TOTAL_FINGER_GAP_UM - FINGER_TIP_GAP_UM) / (2 * TOTAL_FINGER_GAP_UM)  # 0.25 inward margin per side

# empirical buffer on top of the exact physical margin above -- edge_x1/edge_x2
# (the fitted outer-boundary lines) don't always land precisely on the true
# outer metal edge, so the exact-physics FINGER_L_FRACTION math can still land
# a few pixels inside the metal finger's own tip. Confirmed by direct visual
# inspection of a real image (E8): a visibly different, tighter periodic
# pattern -- the metal tip's own serration, not the domain grating -- started
# a handful of pixels inside the calibrated x_right. This is a per-side pixel
# safety margin, not a physical constant -- retune here, not FINGER_L_FRACTION.
# Raised from 0.05 to 0.10 after 0.05 still left visible metal intrusion on
# the right side of several images (D12/D13/D14 mildly, E7/H8 severely).
EDGE_SAFETY_MARGIN_FRACTION = 0.10


MAD_OUTLIER_THRESHOLD = 3.5  # modified z-score cutoff (Iglewicz & Hoaglin) -- standard robust-outlier threshold


def reject_edge_outliers(points, threshold=MAD_OUTLIER_THRESHOLD):
    """Drop points whose x is a robust outlier vs. the rest of this edge's
    detected points, via a MAD-based modified z-score. A single bad row (e.g.
    a border/vignette artifact detect_edge_points latched onto despite
    EDGE_MARGIN_FRACTION) sitting at the extreme top or bottom of the sampled
    rows has outsized leverage on cv2.fitLine's slope -- Huber loss alone
    doesn't reliably discount it when it's only 1-2 points out of ~20."""
    if len(points) < 4:
        return points
    xs = np.array([p[0] for p in points], dtype=float)
    median_x = np.median(xs)
    mad = np.median(np.abs(xs - median_x))
    if mad == 0:
        return points
    modified_z = 0.6745 * (xs - median_x) / mad
    return [p for p, z in zip(points, modified_z) if abs(z) <= threshold]


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


def detect_boundary_tilt(gray, margin_fraction=EDGE_MARGIN_FRACTION, n_samples=N_EDGE_SAMPLES):
    """Detect the two outer finger-comb boundary lines in gray and return
    their averaged, numerically-stable tilt as (dir_x, dir_y) -- same
    approach as detect_roi's own internal computation, factored out
    separately so it can be re-run on an ALREADY-DESKEWED image.

    Why re-run rather than reuse detect_roi's result: the boundary's tilt is
    measured in whatever frame gray is in. detect_roi measures it in the
    original (pre-deskew) image, but compute_duty_cycle sweeps columns on
    the deskewed image -- a rotation composes onto that tilt, so reusing the
    pre-deskew number directly would need converting it through the deskew
    rotation by hand. Measuring fresh on the deskewed image sidesteps that
    composition (and the sign-convention risk that comes with it) entirely."""
    bottom_row_index = gray.shape[0] - 1
    sample_rows = np.linspace(
        int(margin_fraction * bottom_row_index),
        int((1 - margin_fraction) * bottom_row_index),
        n_samples,
        dtype=int,
    )
    left_points = reject_edge_outliers(detect_edge_points(gray, sample_rows, from_right=False))
    right_points = reject_edge_outliers(detect_edge_points(gray, sample_rows, from_right=True))
    if len(left_points) < 2 or len(right_points) < 2:
        raise RuntimeError("Not enough edge points detected to measure boundary tilt.")

    left_arr = np.array(left_points, dtype=np.int32)
    vx, vy, _, _ = cv2.fitLine(left_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vy < 0:
        vx, vy = -vx, -vy

    right_arr = np.array(right_points, dtype=np.int32)
    vx2, vy2, _, _ = cv2.fitLine(right_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vy2 < 0:
        vx2, vy2 = -vx2, -vy2

    avg_vx, avg_vy = (vx + vx2) / 2, (vy + vy2) / 2
    norm = np.hypot(avg_vx, avg_vy)
    return float(avg_vx / norm), float(avg_vy / norm)


def detect_roi(image, roi_height=400, show=True):
    """Detect the two finger edges in image (a path to an image file, or an
    already-loaded numpy array) and return the ROI's slope and x-bounds at the
    top row (y=0), ready to hand to duty_profile.compute_duty_cycle."""

    gray = to_gray(image)

    bottom_row_index = gray.shape[0] - 1
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

    left_points_clean = reject_edge_outliers(left_points)
    right_points_clean = reject_edge_outliers(right_points)
    if len(left_points_clean) < len(left_points):
        print(f"Left edge: dropped {len(left_points) - len(left_points_clean)} outlier point(s) before line fit")
    if len(right_points_clean) < len(right_points):
        print(f"Right edge: dropped {len(right_points) - len(right_points_clean)} outlier point(s) before line fit")
    left_points, right_points = left_points_clean, right_points_clean

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

    FINGER_L = int(round(FINGER_L_FRACTION * (edge_x2 - edge_x1)))
    EDGE_SAFETY_MARGIN = int(round(EDGE_SAFETY_MARGIN_FRACTION * (edge_x2 - edge_x1)))
    print(f"edge_x1={edge_x1}, edge_x2={edge_x2}, FINGER_L={FINGER_L} ({FINGER_L_FRACTION:.2f} of span), "
          f"edge_safety_margin={EDGE_SAFETY_MARGIN} ({EDGE_SAFETY_MARGIN_FRACTION:.2f} of span)")

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

    # avg_m (rise/run = vy/vx) is a reasonable summary slope for a mostly-
    # horizontal line, but these edges are mostly VERTICAL (small grating
    # tilt is the normal case for this device) -- dividing by the near-zero
    # vx there makes slope1/slope2 numerically unstable, amplifying ordinary
    # per-row detection noise into wild, even sign-flipped values (confirmed
    # on a real image: -48.7 vs +59.7 for two edges of the same rigid comb,
    # which should be near-identical). Direction is instead taken straight
    # from the averaged (vx, vy) unit vector -- stable regardless of how
    # close to vertical the line is, since it never divides by either
    # component alone.
    norm = np.hypot(avg_vx, avg_vy)
    dir_x, dir_y = avg_vx / norm, avg_vy / norm

    # x03/x04 are the physically-calibrated ROI edges (see FINGER_L_FRACTION
    # above) -- no contrast-based adjustment. A contrast search here would
    # chase high-contrast regions, which an overpoled area lacks by
    # definition, silently excluding exactly the region that matters most.
    x03, y03 = edge_x1 + FINGER_L + EDGE_SAFETY_MARGIN, bottom_row_index
    dx3 = int(dir_x * length)
    dy3 = int(dir_y * length)

    pt5 = (max(0, x03 - dx3), max(0, y03 - dy3))
    pt6 = (min(img.shape[1] - 1, x03 + dx3), min(img.shape[0] - 1, y03 + dy3))

    cv2.line(img, pt5, pt6, (128, 0, 255), 2)

    x04, y04 = edge_x2 - FINGER_L - EDGE_SAFETY_MARGIN, bottom_row_index
    dx4 = int(dir_x * length)
    dy4 = int(dir_y * length)

    pt7 = (max(0, x04 - dx4), max(0, y04 - dy4))
    pt8 = (min(img.shape[1] - 1, x04 + dx4), min(img.shape[0] - 1, y04 + dy4))

    cv2.line(img, pt7, pt8, (128, 0, 255), 2)

    roi_top_y = 0
    roi_bottom_y = roi_height

    # run per unit rise (dir_x/dir_y), not avg_m's rise-per-run -- see the
    # note above dir_x/dir_y's definition. This extrapolates x03/x04 (anchored
    # at the image bottom) up to roi_top_y over a potentially large number of
    # rows, so an unstable slope here doesn't just add noise, it can shift the
    # whole ROI by tens of pixels (confirmed: ~54px on a real image before
    # this fix).
    inv_slope = dir_x / dir_y
    x03_top = x03 + (roi_top_y - y03) * inv_slope
    x03_bottom = x03 + (roi_bottom_y - y03) * inv_slope
    x04_top = x04 + (roi_top_y - y04) * inv_slope
    x04_bottom = x04 + (roi_bottom_y - y04) * inv_slope

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
        # the physically-calibrated ROI (FINGER_TIP_GAP_UM, centered in the
        # detected span) -- this is the tall region to sweep for measurement.
        "x_left": int(round(x_left)),
        "x_right": int(round(x_right)),
        # full detected boundary span (edge of metal to edge of metal) --
        # reference/debugging only, not safe to sweep: includes the metal
        # fingers themselves.
        "full_x_left": edge_x1,
        "full_x_right": edge_x2,
    }


if __name__ == "__main__":
    detect_roi(DEFAULT_IMAGE, show=True)
