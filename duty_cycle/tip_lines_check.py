import os
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks

ALGORITHMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms")
sys.path.insert(0, ALGORITHMS_DIR)

from loader import to_gray  # noqa: E402
from edge import detect_roi, detect_edge_points, detect_boundary_tilt  # noqa: E402
from duty_profile import (get_manual_angle, deskew, find_tip_alignment_row,  # noqa: E402
                           _find_tip_peak_pairs, interpolate_missing_tips, measurement_window,
                           GAP_OUTLIER_RATIO, MEASUREMENT_HEIGHT, TIP_SEARCH_MARGIN,
                           strip_signal, _valley_edges, MIN_DIP_DEPTH_FRACTION,
                           refine_trough_position)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def draw_tip_lines(image_path, angle=None, upscale=4):
    """Visual ground-truth check for what duty_cycle_for_signal actually uses
    as its period anchors. Draws the raw left/right boundary trace plus the
    tip-to-tip connector for each detected pair, using the SAME detection
    (_find_tip_peak_pairs, imported directly -- not reimplemented, so this
    can't silently drift from what's actually running) the production
    pipeline uses.

    Each connector is drawn between the two sides' OWN individually detected
    tip positions (row_l, x_l) and (row_r, x_r), not forced through a shared
    row -- a deskew angle calibrated from one slice of the image doesn't
    perfectly square up every finger pair across the whole strip, so a real
    tip-to-tip line can have a slight slope; forcing it horizontal would
    hide that instead of showing it.

    Consecutive pairs whose row-spacing is a statistical outlier relative to
    the image's own typical spacing (GAP_OUTLIER_RATIO) are flagged in a
    different color: a real chip's period spacing is approximately
    constant, so an unusually large gap almost always means one or more
    real tip pairs were missed in between, not that domain A is genuinely
    that many times wider there."""
    img = to_gray(image_path)
    if angle is None:
        angle = get_manual_angle(image_path)
        if angle is None:
            raise RuntimeError(f"No manual angle saved for {image_path} -- run deskew_check.py first.")
    deskewed = deskew(img, angle)

    top_row, bottom_search_row = measurement_window(deskewed.shape[0])
    bottom_row = find_tip_alignment_row(deskewed, bottom_search_row)
    lo = max(TIP_SEARCH_MARGIN, top_row)
    hi = min(deskewed.shape[0] - TIP_SEARCH_MARGIN, bottom_row)

    mode, pairs = _find_tip_peak_pairs(deskewed, lo, hi)

    all_rows = np.arange(lo, hi)
    left_points = detect_edge_points(deskewed, all_rows, from_right=False)
    right_points = detect_edge_points(deskewed, all_rows, from_right=True)
    left_by_row = {y: x for x, y in left_points}
    right_by_row = {y: x for x, y in right_points}

    canvas = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    # thin traces of the raw detected boundary on each side, over the full search window
    for row in sorted(left_by_row):
        cv2.circle(canvas, (left_by_row[row], row), 0, (255, 180, 0), -1)
    for row in sorted(right_by_row):
        cv2.circle(canvas, (right_by_row[row], row), 0, (0, 180, 255), -1)

    # the anchor row (single closest-to-center tip alignment), highlighted separately
    cv2.line(canvas, (0, bottom_row), (canvas.shape[1] - 1, bottom_row), (128, 128, 128), 1)

    rows = [p["row"] for p in pairs]
    gaps = np.diff(rows) if len(rows) > 1 else np.array([])
    median_gap = float(np.median(gaps)) if len(gaps) else 0.0
    is_gap_outlier = gaps > GAP_OUTLIER_RATIO * median_gap if len(gaps) else np.array([])

    n_drawn = 0
    if mode == "aligned":
        for p in pairs:
            # yellow if the gap to the NEXT pair is an outlier (this pair's own
            # connector is fine; it's the relationship to its neighbor that's suspect)
            idx = rows.index(p["row"])
            suspect_next = idx < len(is_gap_outlier) and is_gap_outlier[idx]
            color = (0, 255, 255) if suspect_next else (0, 255, 0)
            cv2.line(canvas, (p["x_l"], p["row_l"]), (p["x_r"], p["row_r"]), color, 1)
            n_drawn += 1
    else:
        # single-side fallback: no opposite-side point to pair with, so no honest
        # connector can be drawn -- mark the reliable side's own tip points instead
        color = (255, 180, 0) if mode == "left_only" else (0, 180, 255)
        for p in pairs:
            cv2.drawMarker(canvas, (p["x"], p["row"]), color, cv2.MARKER_TILTED_CROSS, 6, 1)

    n_outliers = int(is_gap_outlier.sum()) if len(is_gap_outlier) else 0
    print(f"{os.path.basename(image_path)}: mode={mode}, {len(pairs)} tip pairs found, "
          f"{n_drawn} connector lines drawn, median gap={median_gap:.1f}px, "
          f"{n_outliers} gap(s) flagged as non-consecutive (>{GAP_OUTLIER_RATIO}x median)")
    if n_outliers:
        for i, g in enumerate(gaps):
            if is_gap_outlier[i]:
                print(f"  gap between row {rows[i]} and row {rows[i+1]}: {g}px "
                      f"(~{g/median_gap:.1f}x median -- likely {round(g/median_gap)-1} missed pair(s))")
    print(f"rows = {rows}")

    # the "made-up" rows: what interpolate_missing_tips inserts into each
    # outlier gap, drawn distinctly (magenta, dashed) since these are an
    # ESTIMATE (assumed evenly spaced), not a detection -- see
    # interpolate_missing_tips's docstring. Drawn using whatever boundary
    # position detect_edge_points found at that exact row (real pixel data),
    # even though the row itself wasn't picked out as a peak.
    interp_rows = interpolate_missing_tips(rows) if mode == "aligned" else []
    made_up_rows = [r for r in interp_rows if r not in rows]
    for row in made_up_rows:
        xl = left_by_row.get(row)
        xr = right_by_row.get(row)
        if xl is None or xr is None:
            continue
        for x in range(xl, xr, 6):  # dashed: draw short segments with gaps
            x_end = min(x + 3, xr)
            cv2.line(canvas, (x, row), (x_end, row), (255, 0, 255), 1)
    print(f"made-up (interpolated) rows = {made_up_rows}")

    # crop to the search window plus a small margin, then upscale -- these
    # images are small enough (~100-150px wide) that 1x is hard to inspect by eye
    y0, y1 = max(0, lo - 10), min(deskewed.shape[0], hi + 10)
    crop = canvas[y0:y1]
    crop = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{stem}_tip_lines.png")
    cv2.imwrite(out_path, crop)
    print(f"Saved: {out_path}")
    return out_path


def draw_domain_widths(image_path, angle=None, upscale=4, frac=0.5, x_spacing=3):
    """Visual ground-truth check for what duty_cycle_for_signal's valley
    measurement actually found: one vertical line per period, drawn at a
    DIFFERENT x position per period (so adjacent periods -- which share a
    row boundary -- don't visually merge into one line), colored green for
    the domain-A portion(s) and red for the domain-B valley in between.
    Uses the exact same period boundaries and sub-pixel valley-edge
    function (_valley_edges) as the production measurement, at the single
    representative threshold frac=0.5 (the middle of the confidence sweep),
    so what's drawn is what the algorithm actually found, not a
    re-derived approximation."""
    img = to_gray(image_path)
    if angle is None:
        angle = get_manual_angle(image_path)
        if angle is None:
            raise RuntimeError(f"No manual angle saved for {image_path} -- run deskew_check.py first.")
    deskewed = deskew(img, angle)
    roi = detect_roi(image_path, roi_height=400, show=False)
    x_left, x_right = roi["x_left"], roi["x_right"]

    top_row, bottom_search_row = measurement_window(deskewed.shape[0])
    bottom_row = find_tip_alignment_row(deskewed, bottom_search_row)
    lo_pairs = max(TIP_SEARCH_MARGIN, top_row)
    hi_pairs = min(deskewed.shape[0] - TIP_SEARCH_MARGIN, bottom_row)
    mode, pairs = _find_tip_peak_pairs(deskewed, lo_pairs, hi_pairs)
    aligned_rows_global = [p["row"] for p in pairs]
    aligned_local = sorted(bottom_row - r for r in aligned_rows_global)

    try:
        dir_x, dir_y = detect_boundary_tilt(deskewed)
        x_per_row = -dir_x / dir_y
    except RuntimeError:
        x_per_row = 0.0

    x0 = (x_left + x_right) // 2
    signal = strip_signal(deskewed, x0, bottom_row, height=MEASUREMENT_HEIGHT, x_per_row=x_per_row)
    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
    min_dip_depth = max(1.0, MIN_DIP_DEPTH_FRACTION * iqr)

    canvas = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    n_periods = len(aligned_local) - 1
    x_start = x0 - (n_periods // 2) * x_spacing

    for i in range(n_periods):
        tip, next_tip = aligned_local[i], aligned_local[i + 1]
        period = next_tip - tip
        if period <= 0:
            continue
        local_signal = signal[tip:next_tip + 1]
        row_tip = bottom_row - tip           # bottom (larger row #) of this period
        row_next_tip = bottom_row - next_tip  # top (smaller row #) of this period
        x_line = x_start + i * x_spacing

        edges = _valley_edges(local_signal, frac, min_dip_depth)
        if edges is None:
            # dip too shallow to represent a real valley -- whole period drawn as domain A
            cv2.line(canvas, (x_line, row_next_tip), (x_line, row_tip), (0, 255, 0), 1)
            continue
        left, right = edges  # local offsets from `tip`, left < right, so `left` is closer to
        # tip (larger absolute row) and `right` is closer to next_tip (smaller absolute row) --
        # i.e. absolute row DECREASES as the local offset increases, since row = bottom_row - local.
        row_valley_near_tip = int(round(bottom_row - (tip + left)))            # bottom edge of valley
        row_valley_near_next_tip = int(round(bottom_row - (tip + right)))      # top edge of valley

        cv2.line(canvas, (x_line, row_next_tip), (x_line, row_valley_near_next_tip), (0, 255, 0), 1)  # A, upper half
        cv2.line(canvas, (x_line, row_valley_near_next_tip), (x_line, row_valley_near_tip), (0, 0, 255), 1)  # B
        cv2.line(canvas, (x_line, row_valley_near_tip), (x_line, row_tip), (0, 255, 0), 1)  # A, lower half

    # keep the tip-to-tip context lines faintly visible for reference -- drawn
    # between each side's OWN individually detected tip point (row_l, x_l) and
    # (row_r, x_r), same as draw_tip_lines, not forced through a shared row.
    # A flat horizontal reference line here would misrepresent the real
    # (slightly non-horizontal) boundary shape, same issue as the connectors.
    if mode == "aligned":
        for p in pairs:
            cv2.line(canvas, (p["x_l"], p["row_l"]), (p["x_r"], p["row_r"]), (90, 90, 90), 1)
    else:
        color = (140, 90, 0) if mode == "left_only" else (0, 90, 140)
        for p in pairs:
            cv2.drawMarker(canvas, (p["x"], p["row"]), color, cv2.MARKER_TILTED_CROSS, 5, 1)

    y0, y1 = max(0, lo_pairs - 10), min(deskewed.shape[0], hi_pairs + 10)
    crop = canvas[y0:y1]
    crop = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{stem}_domain_widths.png")
    cv2.imwrite(out_path, crop)
    print(f"Saved: {out_path}")
    return out_path


def find_combined_trough_row(deskewed, row_lo, row_hi, x_left, x_right, x_per_row, n_samples=15):
    """Find the trough within one gap [row_lo, row_hi] (between two
    consecutive tip-alignment rows), combining independent trough
    detections from MULTIPLE columns across [x_left, x_right] into one
    robust row position via their median -- not trusting a single column's
    measurement, same reasoning as why the tip-to-tip connector uses each
    side's own detected point instead of assuming one row represents both.

    Returns (combined_row, per_column_points) -- per_column_points is a list
    of (x0, row) pairs, one per sampled column, kept at their OWN x0 (not
    collapsed to a single x) so the caller can visualize the actual spread
    of independent detections across the width, not just the combined
    result."""
    height = row_hi - row_lo
    if height < 2:
        return float(row_hi), []
    sample_xs = np.linspace(x_left, x_right, num=min(n_samples, x_right - x_left + 1), dtype=int)
    points = []
    for x0 in sample_xs:
        local = strip_signal(deskewed, x0, row_hi, height=height, x_per_row=x_per_row)
        local_idx = refine_trough_position(local, int(np.argmin(local)))
        points.append((int(x0), row_hi - local_idx))
    return float(np.median([r for _, r in points])), points


def draw_domain_widths_multicolumn(image_path, angle=None, upscale=4, x_spacing=3, n_samples=15):
    """Same intent as draw_domain_widths (show what the algorithm found as
    domain A vs domain B, period by period), but building the A/B boundary
    from find_combined_trough_row's multi-column median instead of a
    single-column threshold crossing -- per the user's original
    description ("go up and down to find where we trough") and the
    explicit correction that a single column (x0) shouldn't stand in for
    the whole ROI width, same lesson as the tip-to-tip connector fix.

    Domain A instance i (centered on tip row i) is bounded by the combined
    trough of the gap BEFORE it and the combined trough of the gap AFTER
    it -- so this only draws A instances that have both neighboring gaps
    available (the first and last tip row in the window are boundary
    cases with only one neighbor, and are skipped)."""
    img = to_gray(image_path)
    if angle is None:
        angle = get_manual_angle(image_path)
        if angle is None:
            raise RuntimeError(f"No manual angle saved for {image_path} -- run deskew_check.py first.")
    deskewed = deskew(img, angle)
    roi = detect_roi(image_path, roi_height=400, show=False)
    x_left, x_right = roi["x_left"], roi["x_right"]

    top_row, bottom_search_row = measurement_window(deskewed.shape[0])
    bottom_row = find_tip_alignment_row(deskewed, bottom_search_row)
    lo_pairs = max(TIP_SEARCH_MARGIN, top_row)
    hi_pairs = min(deskewed.shape[0] - TIP_SEARCH_MARGIN, bottom_row)
    mode, pairs = _find_tip_peak_pairs(deskewed, lo_pairs, hi_pairs)
    detected_rows = [p["row"] for p in pairs]  # ascending: detected_rows[0] is topmost (smallest row)
    tip_rows = interpolate_missing_tips(detected_rows)
    made_up_rows = set(tip_rows) - set(detected_rows)

    try:
        dir_x, dir_y = detect_boundary_tilt(deskewed)
        x_per_row = -dir_x / dir_y
    except RuntimeError:
        x_per_row = 0.0

    canvas = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    # made-up (interpolated) tip rows -- same magenta-dashed treatment as draw_tip_lines,
    # so it's clear which period boundaries are estimated rather than detected
    for row in made_up_rows:
        for x in range(x_left, x_right, 6):
            cv2.line(canvas, (x, row), (min(x + 3, x_right), row), (255, 0, 255), 1)

    # one combined trough per gap between consecutive tip rows
    gap_troughs = []
    for i in range(len(tip_rows) - 1):
        row_hi, row_lo = tip_rows[i + 1], tip_rows[i]  # row_hi > row_lo (ascending rows go down the image)
        combined, points = find_combined_trough_row(deskewed, row_lo, row_hi, x_left, x_right, x_per_row, n_samples)
        gap_troughs.append(combined)
        # each column's OWN independent trough detection, at its OWN x0 -- this is the
        # "multiple points across the x-axis" that get combined, not one stand-in point
        for x0, r in points:
            cv2.circle(canvas, (x0, int(round(r))), 1, (80, 80, 220), -1)
        # the combined (median) trough row, drawn across the ROI width for reference
        cv2.line(canvas, (x_left, int(round(combined))), (x_right, int(round(combined))), (255, 0, 255), 1)

    n_periods = len(tip_rows) - 2  # A instances with both neighbors available
    if n_periods > 0:
        x0 = (x_left + x_right) // 2
        x_start = x0 - (n_periods // 2) * x_spacing
        for i in range(n_periods):
            row_tip = tip_rows[i + 1]
            trough_before, trough_after = gap_troughs[i], gap_troughs[i + 1]  # after=smaller row (up), before=larger row (down)
            x_line = x_start + i * x_spacing
            cv2.line(canvas, (x_line, row_tip), (x_line, int(round(trough_after))), (0, 255, 0), 1)
            cv2.line(canvas, (x_line, int(round(trough_before))), (x_line, row_tip), (0, 255, 0), 1)

    y0, y1 = max(0, lo_pairs - 10), min(deskewed.shape[0], hi_pairs + 10)
    crop = canvas[y0:y1]
    crop = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{stem}_domain_widths_multicol.png")
    cv2.imwrite(out_path, crop)
    print(f"Saved: {out_path}")
    print(f"tip_rows={tip_rows}")
    print(f"gap_troughs (combined, median of {n_samples} columns)={[round(t,1) for t in gap_troughs]}")
    return out_path


OUTLIER_MAD_MULTIPLIER = 2.5  # points whose trough row deviates from the local median by more than this
# many MADs are rejected as likely shadow artifacts -- the user's own observation: nearby columns should
# give a similar duty cycle (physical continuity), so a wildly different point is more likely the darkest
# pixel landing on the domain material itself (shadowed regions) rather than the true A/B boundary, not a
# genuine different measurement.
OUTLIER_MIN_THRESHOLD_PX = 1.5  # floor on the rejection threshold, so near-zero MAD (all points already
# tightly agreeing) doesn't make the check hypersensitive to ordinary sub-pixel jitter.


def _tip_baseline_fn(pair, x_per_row):
    """The tip-to-tip CONNECTING LINE for one tip, as a function x0 ->
    expected row at that column -- the baseline of domain A, per the
    user's explicit correction (a single shared tip row is exactly the
    same "force it horizontal" mistake already fixed once for the
    connector line itself; the fix has to carry through to this search
    too, not stop at the drawing).

    For an aligned-mode pair, fits the real line through its own two
    independently-detected points (x_l, row_l) and (x_r, row_r) -- the
    same two points draw_tip_lines connects. For a fallback-mode pair
    (only one side has a usable point) or an interpolated row (no
    detected point at all), there's no second point to fit a real line
    from, so this falls back to a flat line at that single row -- x_per_row
    describes the metal boundary's OWN edge slant (used for slanting the
    brightness strip extraction), not how the tip-alignment row shifts
    with column, so it can't be repurposed here via a simple inversion
    (confirmed: doing that gave a wildly wrong, huge slope and crashed by
    indexing rows outside the image)."""
    if "x_l" in pair:
        x_l, row_l, x_r, row_r = pair["x_l"], pair["row_l"], pair["x_r"], pair["row_r"]
        m = (row_r - row_l) / (x_r - x_l) if x_r != x_l else 0.0
        b = row_l - m * x_l
    else:
        m, b = 0.0, float(pair["row"])
    return lambda x: m * x + b


MIN_BOUNDARY_CONTRAST = 30  # a column-level admission gate: is there ANY plausibly-real dark
# feature in this window at all. Was 100, calibrated against the brighter part of the image --
# confirmed wrong on real data (E10): the bottom ~150px of the image is genuinely dimmer overall
# (baseline ~90-120 vs ~150-180 elsewhere), so a real, consistent boundary there never exceeded
# ~87 contrast and was rejected outright (0/60 points found) no matter how many columns were
# sampled. The real discrimination between genuine boundaries and noise now happens per-candidate
# via MIN_TROUGH_DEPTH below, not at this whole-column level -- this just filters out flat/noise
# columns before that finer check runs.

MIN_TROUGH_DEPTH = 40  # a candidate trough must be at least this much darker than the window's
# baseline to be eligible for the nearest-to-tip tiebreak below. Without this, relaxing
# MIN_BOUNDARY_CONTRAST let the tiebreak pick shallow noise blips (depth <20px, sometimes even
# negative -- brighter than baseline) sitting closer to the tip than the real, deeper trough
# further out -- confirmed on real data (E10, tip42): per-column dumps showed the "nearest"
# candidate at depth -8 to 46 while a genuine trough at depth 56-87.5 sat a few pixels further
# away. Filtering candidates by depth first, then picking nearest among survivors, recovers the
# real trough without reopening the earlier fabricated-boundary problem (confirmed visually on
# the full E10 image after this fix: every magenta boundary line landed on a real, correct seam,
# including the three periods previously flagged as fabricated).
#
# Was 60 until real data from a different microscope (LNLaser03 page6) showed it was too high:
# a tip's own correct, nearby trough measured depth ~55 (just under 60) and got filtered out,
# leaving only a genuine-but-wrong, farther trough (depth ~67, belonging to the NEXT tip) as the
# sole survivor -- so the nearest-to-tip tiebreak never got a chance to run, since there was only
# one candidate left. Lowered to 40 after confirming E10 and G7 are completely unaffected by any
# threshold from 35-60 (identical valid-period sets throughout that whole range -- their real
# transitions are comfortably deeper), while page6 recovers from 8 to 22 correct domain-A
# instances (essentially every tip that isn't a structural edge case) once it drops to 40 or below.

TROUGH_FLANK_PX = 3  # a candidate trough must be strictly darker than the signal at least this many
# pixels out on BOTH sides -- find_peaks' prominence check alone can still pass a point that's only
# darker than its immediate neighbor (e.g. one noisy pixel next to a slightly-less-noisy one), without
# it being part of a genuine, sustained dip. Confirmed the user's concern is real: prominence filtering
# rejects low-CONTRAST noise but doesn't check the dip actually has real WIDTH on both sides.

TROUGH_FLANK_TOLERANCE = 5  # px of slack allowed in the flank check above -- requiring flank pixels
# to be STRICTLY darker with zero tolerance made the check too fragile: confirmed on real data
# (LNLaser03 page3, tip6): the correct, nearby trough (depth 68-94, clearly real and always the
# closer of two candidates) intermittently failed the strict check at ~38% of columns, purely from
# single-pixel noise breaking perfect monotonicity within the flank window -- with the ONLY other
# candidate being a real but wrong, farther trough (actually the neighboring tip's own boundary),
# the per-column pick flip-flopped between them, and the resulting scattered line got correctly
# rejected by the scatter check, throwing out a real boundary along with the noise. Confirmed zero
# effect on E10/G7 (identical valid-period sets at tolerance 0, 3, 5, and 8).


def _is_genuine_trough(local, idx, flank=TROUGH_FLANK_PX, tolerance=TROUGH_FLANK_TOLERANCE):
    lo = max(0, idx - flank)
    hi = min(len(local), idx + flank + 1)
    left = local[lo:idx]
    right = local[idx + 1:hi]
    threshold = local[idx] - tolerance
    return (len(left) == 0 or np.all(left > threshold)) and (len(right) == 0 or np.all(right > threshold))


def _find_boundary_points(deskewed, tip_fn, direction, bound_fn, x_left, x_right, x_per_row, n_samples=15):
    """For ONE tip (given as its baseline line, tip_fn), search in
    `direction` ('up' or 'down') for the nearest trough, independently at
    multiple columns across [x_left, x_right] -- the raw (unfiltered)
    points this domain-A instance's boundary line gets fit from. At each
    column, the search starts from tip_fn's OWN row at that column (not a
    single shared row for every column) and is bounded by bound_fn (the
    next tip's baseline, also evaluated at that same column) so it can't
    cross into the neighboring period."""
    sample_xs = np.linspace(x_left, x_right, num=min(n_samples, x_right - x_left + 1), dtype=int)
    points = []
    for x0 in sample_xs:
        tip_row = tip_fn(x0)
        bound_row = bound_fn(x0)
        row_hi, row_lo = (tip_row, bound_row) if direction == "up" else (bound_row, tip_row)
        row_hi, row_lo = int(round(row_hi)), int(round(row_lo))
        height = row_hi - row_lo
        if height < 2:
            continue
        local = strip_signal(deskewed, x0, row_hi, height=height, x_per_row=x_per_row)
        # index 0 of `local` is always row_hi -- for "up" that's tip_row (the search's actual
        # starting point), for "down" that's bound_row (the far end), since strip_signal always
        # indexes from whichever reference row it's given, regardless of search direction.
        start_idx = 0 if direction == "up" else len(local) - 1
        # prominence=2 (a fixed, low threshold) caught small noise-level wiggles right next to
        # the tip alongside the real boundary trough -- confirmed on real data (G7, x0=530):
        # prominence 2 found 7 candidate "troughs" including several with prominence 5-17 (noise)
        # sitting closer to the tip than the two real ones (prominence 103 and 172), so "nearest
        # to the tip" picked noise. Scaling to the column's own dynamic range filters those out
        # while still finding real troughs regardless of the image's absolute brightness/contrast.
        # A real domain A/B boundary line is a genuinely dark seam, not just "darker than its own
        # local prominence" -- confirmed on real data (E10): comparing baseline (this window's own
        # endpoint brightness -- domain A on both sides, by construction) against the deepest point
        # found, confirmed-real boundaries showed contrast 124-175, while two of three flagged-wrong
        # cases showed only ~76-77, roughly half. This alone doesn't catch every wrong case (one
        # flagged case, a real base-material edge showing through where no domain A ever formed,
        # happened to be just as dark as a real boundary), but it's an independent signal from the
        # per-column scatter check and catches cases scatter alone doesn't.
        #
        # baseline uses the window's own 90th percentile, not the two endpoint pixels -- confirmed
        # wrong on real data from a different microscope/lighting setup (LNLaser03): the tip itself
        # sat measurably dimmer (~25-40px) than the true bright domain plateau just beyond it
        # (shadow at the metal edge, not a brightness/exposure problem -- a global contrast stretch
        # wouldn't fix it), so every contrast check kept failing despite the real dip being 55-70px
        # deep relative to the actual plateau. The 90th percentile finds that true plateau directly
        # regardless of whether the endpoints happen to sit on it. Verified to change nothing on
        # already-validated data: identical valid-boundary counts on E10 (84/84) and G7 (92/92)
        # before and after this change, while recovering 14 previously-unfindable boundaries here.
        baseline = float(np.percentile(local, 90))
        contrast = baseline - float(local.min())
        if contrast < MIN_BOUNDARY_CONTRAST:
            continue
        prominence = max(10.0, 0.1 * (float(local.max()) - float(local.min())))
        troughs, _ = find_peaks(-local, prominence=prominence)
        troughs = np.array([t for t in troughs if _is_genuine_trough(local, int(t))], dtype=int)
        # per-candidate depth floor, on top of the flank check -- see MIN_TROUGH_DEPTH. A trough
        # can be locally genuine (flanked by strictly brighter pixels within TROUGH_FLANK_PX) while
        # still just being noise right next to the tip, nowhere near as dark as a real boundary.
        troughs = np.array([t for t in troughs if (baseline - float(local[t])) >= MIN_TROUGH_DEPTH], dtype=int)
        if len(troughs) == 0:
            # No genuine trough anywhere in this window -- do NOT fabricate one by falling back
            # to the single darkest pixel. Confirmed by the user: for underpoled samples, some
            # metal gaps genuinely have no domain A at all (the whole gap is domain B), so there
            # is no real A/B boundary to find here. Silently grabbing argmin regardless produced
            # a fake boundary line and a nonsensical measurement for exactly these periods --
            # skipping the column instead lets the caller see "no data" and handle it explicitly.
            continue
        # nearest candidate trough to the search's actual starting tip, not the globally
        # darkest point anywhere in the window -- confirmed on real data (G7): a shallow,
        # correct trough right next to the tip was being skipped in favor of a deeper, wrong
        # one further away (spilling into the ADJACENT period's own boundary), which inflated
        # domain A's measured width past the period itself -- a physically impossible >100%
        # duty cycle, not just an extreme-but-real overpoled reading.
        raw_idx = int(troughs[np.argmin(np.abs(troughs - start_idx))])
        local_idx = refine_trough_position(local, raw_idx)
        points.append((int(x0), row_hi - local_idx))
    return points


RAW_SCATTER_MAX_MAD = 3.8  # px -- above this, the per-column points are too inconsistent with each
# other to represent a real boundary; see the note in _fit_boundary_line. On a full 42-period real
# image (E10), the MAD distribution clustered at 1.2-3.6px for most periods, then jumped to 4.1-9.2px
# for a distinct minority -- 3.8 sits in that actual gap, not just a guessed round number.

MIN_GENUINE_TROUGH_FRACTION = 0.3  # a period whose "up" or "down" search found genuine troughs at
# fewer than this fraction of attempted columns is treated as having NO real boundary at all here, not
# a boundary that's just hard to measure. Confirmed physically real: for underpoled samples, some metal
# gaps have no domain A at all (the whole gap is domain B), so most/all columns correctly find nothing
# -- a handful of stray points passing the flank/prominence check by chance shouldn't be enough to fit
# a fabricated line through.

MAX_HEIGHT_FRACTION = 0.85  # a domain-A instance can occupy at most this fraction of the local
# tip-to-tip period -- domain B must keep some nonzero share. See the comment in
# compute_domain_boundaries for why this replaced a median-of-measured-heights comparison.


def _fit_boundary_line(points, n_attempted=None):
    """Reject outlier points (shadow artifacts -- confirmed by the user's
    own inspection that some columns' darkest pixel lands on the domain
    material rather than the true A/B boundary) via a MAD-based check
    against the local median, then fit a line (row = m*x + b) through the
    survivors. Falls back to a flat line at the median if too few points
    survive to fit -- still allows for the same small physical tilt the
    tip-to-tip connector shows, rather than assuming a perfectly
    horizontal boundary.

    n_attempted (optional): how many columns were actually searched (before
    _find_boundary_points dropped ones with no genuine trough). If given and
    len(points) is too small a fraction of it, returns None -- see
    MIN_GENUINE_TROUGH_FRACTION."""
    if not points:
        return None
    if n_attempted and len(points) < MIN_GENUINE_TROUGH_FRACTION * n_attempted:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    rows = np.array([p[1] for p in points], dtype=float)
    median = np.median(rows)
    mad = np.median(np.abs(rows - median))
    # the RAW scatter (before any outlier rejection) is itself the signal for "is there a real
    # boundary here at all" -- confirmed by the user's own observation: a genuine A/B boundary's
    # per-column points cluster tightly (MAD ~1.5-2.9px on real data), while a period with no
    # domain A at all (the whole gap is domain B, confirmed physically real for underpoled
    # samples) has no consistent feature to track, so each column's "nearest genuine-looking dip"
    # lands somewhere different -- MAD ~4.9-8.0px on the same real data, 3-4x higher. Rejecting
    # individual far-outliers (below) isn't enough by itself: a line fit through whatever's left
    # of a fundamentally inconsistent set of points is still a fabricated boundary, not a
    # measurement, so the whole result is discarded here rather than partially trusted.
    if mad > RAW_SCATTER_MAX_MAD:
        return None
    threshold = max(OUTLIER_MIN_THRESHOLD_PX, OUTLIER_MAD_MULTIPLIER * mad)
    inlier_mask = np.abs(rows - median) <= threshold

    inliers = [(int(x), float(r)) for x, r, keep in zip(xs, rows, inlier_mask) if keep]
    outliers = [(int(x), float(r)) for x, r, keep in zip(xs, rows, inlier_mask) if not keep]

    if inlier_mask.sum() >= 2 and len(set(xs[inlier_mask])) >= 2:
        m, b = np.polyfit(xs[inlier_mask], rows[inlier_mask], 1)
    else:
        m, b = 0.0, float(np.median(rows[inlier_mask])) if inlier_mask.sum() else float(median)

    return {"m": float(m), "b": float(b), "inliers": inliers, "outliers": outliers}


def _eval_line(line, x):
    return line["m"] * x + line["b"]


GAP_AGREEMENT_MAX_PX = 4.0  # px -- how far apart the "down from the earlier tip" and "up from the
# later tip" boundary estimates for the SAME physical gap are allowed to be before treating them as
# disagreeing (and rejecting the gap entirely). Confirmed necessary on real data: the identical gap
# gave MAD=5.02 (correctly rejected) searching one direction and MAD=1.06 (incorrectly ACCEPTED)
# searching the other -- each direction's own internal scatter check isn't enough by itself, since a
# fabricated boundary can look perfectly self-consistent from one side while being obvious garbage
# from the other. Requiring both independent directions to agree catches what either one alone missed.


def _find_gap_boundary(deskewed, tip_fn_before, tip_fn_after, x_left, x_right, x_per_row, n_samples=60):
    """The one real A/B boundary within a single gap between two consecutive tips, cross-validated
    by searching from BOTH directions -- down from the earlier tip, and up from the later tip --
    since they search the identical physical row range and should find the same answer if a real
    boundary exists there. Returns None if either direction's own search is rejected (see
    _fit_boundary_line) OR if the two directions disagree by more than GAP_AGREEMENT_MAX_PX, not just
    when one direction alone looks inconsistent."""
    pts_down = _find_boundary_points(deskewed, tip_fn_before, "down", tip_fn_after, x_left, x_right, x_per_row, n_samples)
    pts_up = _find_boundary_points(deskewed, tip_fn_after, "up", tip_fn_before, x_left, x_right, x_per_row, n_samples)
    line_down = _fit_boundary_line(pts_down, n_attempted=n_samples)
    line_up = _fit_boundary_line(pts_up, n_attempted=n_samples)
    if line_down is None or line_up is None:
        return None
    x_mid = (x_left + x_right) / 2
    if abs(_eval_line(line_down, x_mid) - _eval_line(line_up, x_mid)) > GAP_AGREEMENT_MAX_PX:
        return None
    return _fit_boundary_line(pts_down + pts_up, n_attempted=2 * n_samples)


MIN_TIP_GAP_RATIO = 0.65  # the complementary check to GAP_OUTLIER_RATIO: a consecutive tip gap
# below this fraction of the image's own typical spacing means the LATER tip is almost certainly
# a phantom, not a genuinely denser region -- confirmed on real data (LNLaser03 page4, in
# "left_only" fallback mode where there's no opposite-side alignment to cross-check against): a
# noisy left-edge signal produced an extra peak almost exactly halfway between two real teeth,
# repeatedly, giving gaps of ~20-33px interleaved with the real ~59-62px spacing -- a clean,
# wide gap between the two populations, so 0.65 sits well clear of both (max real-phantom ratio
# ~33/62=0.53, min genuine ratio ~59/62=0.95). Confirmed zero effect on E10/G7, where every gap
# already sits close to the median with no such bimodal split.


def _dedupe_close_tips(rows, min_gap_ratio=MIN_TIP_GAP_RATIO):
    """Drop tips that sit suspiciously close to the previous KEPT tip -- almost certainly a
    phantom secondary peak, not a real, denser-than-usual tooth. Greedy left-to-right instead of
    pairwise: a run of several consecutive phantoms (confirmed on real data, page4: four small
    gaps in a row) needs each one compared against the last REAL tip, not against its own
    immediately preceding phantom, or the whole run would be kept as "close to its neighbor"."""
    rows = sorted(rows)
    if len(rows) < 3:
        return rows
    typical = float(np.median(np.diff(rows)))
    kept = [rows[0]]
    for r in rows[1:]:
        if (r - kept[-1]) < min_gap_ratio * typical:
            continue
        kept.append(r)
    return kept


OFFCENTER_SUM_TOLERANCE = 0.15  # a tip is a repositioning candidate when its two neighbor gaps
# sum to within this fraction of exactly 2x the typical period -- i.e. the two real, correctly-
# spaced teeth on either side of it are right where expected, and only the tip BETWEEN them is
# off.
OFFCENTER_MIN_SHIFT = 0.2  # ...and only actually gets repositioned if it's off the midpoint of
# those two neighbors by at least this fraction of the typical period, so a tip that's already
# essentially centered (just normal per-tooth jitter) is left alone. Confirmed on real data
# (LNLaser03 page4): one real tooth's own peak-detected position (row 707) sat 45px from the
# tip before it and 79px from the tip after it -- individually inconsistent with the ~61px
# typical spacing, but summing to 124 (almost exactly 2x61), meaning the two NEIGHBORS were
# correctly placed and this tip's own detected position was just off-center within that span
# (true position ~724, a 17px correction) -- most likely a secondary bump in the noisy
# "left_only" fallback signal outweighing the real peak by a small margin. Confirmed zero effect
# on E10/G7, whose consistently even spacing never triggers this pattern.


def _reposition_offcenter_tips(rows, sum_tolerance=OFFCENTER_SUM_TOLERANCE, min_shift=OFFCENTER_MIN_SHIFT):
    """A real (not phantom -- see _dedupe_close_tips) tip whose own detected position drifted
    within an otherwise-correct span gets snapped back to the midpoint of its two neighbors."""
    rows = list(rows)
    if len(rows) < 3:
        return rows
    typical = float(np.median(np.diff(rows)))
    for i in range(1, len(rows) - 1):
        combined = rows[i + 1] - rows[i - 1]
        if abs(combined - 2 * typical) > sum_tolerance * typical:
            continue
        midpoint = (rows[i - 1] + rows[i + 1]) / 2
        if abs(rows[i] - midpoint) > min_shift * typical:
            rows[i] = int(round(midpoint))
    return rows


def compute_domain_boundaries(image_path, angle=None, n_samples=60):
    """Shared computation behind both draw_domain_boundary_lines() and duty_cycle_statistics() --
    kept as one function so the picture you look at and the numbers the optimizer sees can never
    drift out of sync with each other.

    For each domain-A instance (each tip row), independently search up (toward the previous tip)
    and down (toward the next tip) for the nearest trough, at multiple columns across the ROI
    width. Reject outlier columns (shadow artifacts) and fit a line through the rest -- this line,
    not a single flat row, is the domain A/B boundary, since it can carry the same small physical
    tilt the tip-to-tip connector line already does.

    Enforces the A/B/A/B alternation: a domain-A instance is defined by ITS OWN tip's up-search and
    down-search (the "same metal connecting line" the two boundaries were found from) -- never a
    mix of two different tips' searches. When a period's lower boundary and its neighbor's upper
    boundary land on the same physical line (shared, not just close), that one line can back only
    ONE of the two competing domain-A claims, not both.

    Which one is real: confirmed on real data (E10) that the fake claimant's total height comes out
    roughly double the real, typical period width, because its search had no genuine nearby
    boundary of its own, so it reached all the way past the shared line to whatever its bound
    function allowed -- e.g. tip10 measured 56px (vs the image's ~26.5px median) by reaching for
    tip11's own upper boundary as if it were tip10's lower boundary, while tip11's own pairing
    (using that identical shared line as ITS upper boundary) came out a normal 26.3px. So for each
    shared line, keep the side whose resulting height is closer to the image's median period
    height, and drop the other -- the wide one is reaching for a boundary that isn't really its own.

    Returns a dict: deskewed, x_left, x_right, lo_pairs, hi_pairs, tip_rows, made_up_rows,
    x_per_row, upper_lines, lower_lines, period_indices (validated, non-shared domain-A instances)."""
    img = to_gray(image_path)
    if angle is None:
        angle = get_manual_angle(image_path)
        if angle is None:
            raise RuntimeError(f"No manual angle saved for {image_path} -- run deskew_check.py first.")
    deskewed = deskew(img, angle)
    roi = detect_roi(image_path, roi_height=400, show=False)
    x_left, x_right = roi["x_left"], roi["x_right"]

    # search the full image height (minus just TIP_SEARCH_MARGIN, the per-side buffer
    # _find_tip_peak_pairs itself needs), not the tighter measurement_window() 5% margin --
    # confirmed on real data (E10) that margin was silently cutting off real teeth: the search
    # stopped at row 1949 while the real comb pattern continues, sharp and undistorted, all the
    # way to the image's bottom edge (row 2048), and widening recovered 2 more genuine, evenly
    # spaced tips (at the established ~42px period) beyond the old cutoff. This matters beyond
    # just counting more periods -- the LAST tip inside the old cutoff had no "next tip" to search
    # its own lower boundary against at all, so it could never pass the paired-boundary check
    # no matter how good the underlying signal was.
    lo_pairs = TIP_SEARCH_MARGIN
    hi_pairs = deskewed.shape[0] - TIP_SEARCH_MARGIN
    mode, pairs = _find_tip_peak_pairs(deskewed, lo_pairs, hi_pairs)
    deduped_rows = _dedupe_close_tips([p["row"] for p in pairs])
    pairs = [p for p in pairs if p["row"] in set(deduped_rows)]
    detected_rows = _reposition_offcenter_tips(deduped_rows)
    # a repositioned tip's pair record needs to move with it (row, and row_l/row_r by the same
    # shift, keeping the real L/R connector slope rather than discarding it) or the row-keyed
    # lookup below would treat this real, just-corrected tip as if no real detection existed at
    # its new row at all.
    for old_row, new_row in zip(deduped_rows, detected_rows):
        if old_row == new_row:
            continue
        p = next(p for p in pairs if p["row"] == old_row)
        delta = new_row - old_row
        p["row"] = new_row
        if "row_l" in p:
            p["row_l"] += delta
        if "row_r" in p:
            p["row_r"] += delta
    tip_rows = interpolate_missing_tips(detected_rows)
    made_up_rows = set(tip_rows) - set(detected_rows)

    try:
        dir_x, dir_y = detect_boundary_tilt(deskewed)
        x_per_row = -dir_x / dir_y
    except RuntimeError:
        x_per_row = 0.0

    # one baseline function per tip row -- the real connecting line (through its own two
    # independently-detected points) for a detected row, or a flat line at that row for an
    # interpolated (made-up) row, which has no real detected point to build a line from
    pair_by_row = {p["row"]: p for p in pairs}
    tip_fns = []
    for tip_row in tip_rows:
        if tip_row in pair_by_row:
            tip_fns.append(_tip_baseline_fn(pair_by_row[tip_row], x_per_row))
        else:
            tip_fns.append(_tip_baseline_fn({"x": (x_left + x_right) // 2, "row": tip_row}, 0.0))

    # Each tip's own up-search and down-search are genuinely INDEPENDENT boundaries, not two
    # measurements of the same point -- confirmed by the user: when domain B has real width, a
    # gap contains the upper domain-A's own lower edge AND the lower domain-A's own upper edge,
    # two different physical boundaries, not one boundary approached from two directions. Forcing
    # them to agree (an earlier version of this code) was wrong for exactly that reason.
    upper_lines, lower_lines = {}, {}
    for i, tip_row in enumerate(tip_rows):
        if i > 0:
            pts = _find_boundary_points(deskewed, tip_fns[i], "up", tip_fns[i - 1], x_left, x_right, x_per_row, n_samples)
            upper_lines[i] = _fit_boundary_line(pts, n_attempted=n_samples)
        if i < len(tip_rows) - 1:
            pts = _find_boundary_points(deskewed, tip_fns[i], "down", tip_fns[i + 1], x_left, x_right, x_per_row, n_samples)
            lower_lines[i] = _fit_boundary_line(pts, n_attempted=n_samples)

    x_anchor_ref = (x_left + x_right) / 2

    def _height(i):
        if upper_lines.get(i) is None or lower_lines.get(i) is None:
            return None
        return _eval_line(lower_lines[i], x_anchor_ref) - _eval_line(upper_lines[i], x_anchor_ref)

    heights = {i: h for i in range(len(tip_rows)) if (h := _height(i)) is not None}

    # A domain-A instance's own height can never plausibly reach its own full period -- there has
    # to be SOME room left for domain B. tip-to-tip spacing is a robust, independent reference for
    # "one period" here (it comes from metal-edge geometry, not from the boundary search being
    # judged), and it's typically rock-steady across a whole image even when the boundary search
    # isn't. Confirmed wrong on real data (LNLaser03 page6): comparing each height against the
    # MEDIAN OF THE OTHER MEASURED HEIGHTS (an earlier version of this check) broke down because
    # the "reaching too far, grabbing the neighbor's boundary instead of its own" failure was the
    # MAJORITY case in that image (14 of 22), so the median itself came out close to the wrong
    # value, making "keep whichever is closer to the median" little better than a coin flip among
    # the bad cluster, and it wrongly penalized the genuinely correct minority. Comparing against
    # tip spacing instead fixed it (correctly kept exactly the true ~30px cluster, dropped every
    # ~58px reaching-too-far one) with zero change on E10/G7 (identical valid-period sets before
    # and after, on both).
    tip_diffs = np.diff(tip_rows)
    typical_period = float(np.median(tip_diffs)) if len(tip_diffs) else 0.0
    max_height = MAX_HEIGHT_FRACTION * typical_period if typical_period else float("inf")

    # upper_lines[i]/lower_lines[i] can now be None (a real, meaningful result -- no genuine
    # domain-A boundary was found there, e.g. an underpoled sample's metal gap with no A at all),
    # not just absent from the dict -- both cases mean "skip this period," so check the value.
    period_indices = [i for i in range(len(tip_rows))
                       if upper_lines.get(i) is not None and lower_lines.get(i) is not None
                       and heights[i] <= max_height]

    return dict(deskewed=deskewed, x_left=x_left, x_right=x_right, lo_pairs=lo_pairs, hi_pairs=hi_pairs,
                tip_rows=tip_rows, made_up_rows=made_up_rows, x_per_row=x_per_row, tip_fns=tip_fns,
                upper_lines=upper_lines, lower_lines=lower_lines, period_indices=period_indices)


def duty_cycle_statistics(image_path, angle=None, n_samples=60):
    """Duty cycle for each validated domain-A instance, classified as NORMAL (immediately adjacent
    to the previous validated instance) or an UNDERPOLED-REGION measurement (its span skipped over
    one or more instances that failed the pairing check in compute_domain_boundaries -- i.e. a
    local stretch where domain A didn't form).

    Each period's width is the RAW distance back to the nearest validated previous instance's own
    lower boundary -- never divided by however many instances were skipped over. Confirmed by the
    user on real data (E10, period 18, skipping tip17): dividing by the skip count smooths an
    abnormally long, underpoled domain B into looking like a normal-length period, hiding exactly
    what duty cycle is supposed to reveal. The raw span IS this domain-A's true period; a long B
    should read as a low duty cycle for that specific measurement, not get averaged away.

    Mixing NORMAL and underpoled-region measurements into one mean is itself misleading -- e.g. on
    E10, blending 12 normal periods (mean 60.6%) with 14 underpoled-region ones (each ~22-33%)
    produces an overall mean of ~44%, which reads as "mildly underpoled" when the sample is
    actually "mostly overpoled, with just over half its measured periods sitting in severely
    underpoled local stretches" -- two distinct problems that would call for different parameter
    adjustments. So they're kept and reported separately: `mean_duty_cycle`/`std_duty_cycle` are
    computed from NORMAL periods only (the honest "how far off-target is the typical region"
    signal), and `problem_fraction` (count of underpoled-region periods / total validated periods)
    is the separate "how much of the sample is locally failing to pole at all" signal -- this maps
    directly onto the existing BO objective's `|mean duty cycle - 0.5| + lambda_1*std +
    lambda_2*problem_fraction` structure, which already expected a term exactly like this.

    Returns a dict: mean_duty_cycle, std_duty_cycle (percent, NORMAL periods only), problem_fraction
    (0-1), n_normal, n_underpoled_region, n_domain_a_found (raw count of validated domain-A
    instances from compute_domain_boundaries, before the duty-cycle-specific "needs a previous
    valid instance" filtering -- 0 or 1 here means no boundary could be found ANYWHERE in the
    image, not just that duty cycle couldn't be computed from what was found), periods (per-period
    list of dicts with index, a_width, period_width, duty_cycle_pct, n_steps, is_normal)."""
    data = compute_domain_boundaries(image_path, angle=angle, n_samples=n_samples)
    upper_lines, lower_lines = data["upper_lines"], data["lower_lines"]
    period_indices = data["period_indices"]
    x_anchor = (data["x_left"] + data["x_right"]) / 2

    periods = []
    for idx_pos, i in enumerate(period_indices):
        if idx_pos == 0:
            continue  # no earlier validated instance to measure a period against
        j = period_indices[idx_pos - 1]
        lo_i = _eval_line(lower_lines[i], x_anchor)
        up_i = _eval_line(upper_lines[i], x_anchor)
        lo_j = _eval_line(lower_lines[j], x_anchor)
        a_width = lo_i - up_i
        period_width = lo_i - lo_j
        if a_width <= 0 or period_width <= 0:
            continue
        n_steps = i - j
        periods.append(dict(index=i, prev_index=j, a_width=a_width, period_width=period_width,
                             duty_cycle_pct=100.0 * a_width / period_width, n_steps=n_steps,
                             is_normal=(n_steps == 1)))

    normal = [p["duty_cycle_pct"] for p in periods if p["is_normal"]]
    n_underpoled = sum(1 for p in periods if not p["is_normal"])
    total = len(periods)

    return dict(
        mean_duty_cycle=float(np.mean(normal)) if normal else None,
        std_duty_cycle=float(np.std(normal)) if normal else None,
        problem_fraction=(n_underpoled / total) if total else None,
        n_normal=len(normal),
        n_underpoled_region=n_underpoled,
        n_domain_a_found=len(period_indices),
        periods=periods,
    )


def estimate_duty_cycle(image_path, angle=None, n_samples=60):
    """Drop-in replacement for gp_trial's estimate_duty_cycle (previously
    duty_cycle/algorithms/main.main, the older column-sweep architecture) --
    same (mean, std, diagnostics) return shape, built on duty_cycle_statistics()
    instead. angle resolution: an explicit manual angle if one was saved via
    deskew_check.py, otherwise 0.0 -- NOT automatic angle detection. Confirmed
    earlier this session that automatic detection (find_deskew_angle) fails on
    roughly a third of real images, while the batch this pipeline currently
    runs on (ZEHRA01) was confirmed by the user to be already near-horizontal,
    so 0.0 is the safer default here, not a guess.

    diagnostics keeps the same "problem_fraction" and "classification" keys
    main.py/bo_session.py already read (classification is "spatially_varying"
    when problem_fraction > 0, else "ok", to trigger the existing CLI warning),
    plus the full underlying stats dict under "duty_cycle_stats" for anything
    that wants the finer per-period breakdown.

    Returns (None, None, diagnostics) when no numeric duty cycle could be computed, with
    classification distinguishing WHY: "severely_overpoled" when compute_domain_boundaries found
    0 or 1 validated domain-A instances anywhere in the image (confirmed on real data, F9: domain A
    had merged into its neighbors so completely that no genuine A/B transition existed anywhere to
    detect -- tip geometry itself was fine, 61 tip rows found, but 0 valid boundary pairs) --
    that's a real, meaningful result (too overpoled to even measure), not a technical failure.
    "unmeasurable" is reserved for the rarer case where several domain-A instances WERE found but
    still not enough to compute even one duty-cycle period from (e.g. all isolated, none adjacent
    enough in sequence)."""
    if angle is None:
        angle = get_manual_angle(image_path) or 0.0
    stats = duty_cycle_statistics(image_path, angle=angle, n_samples=n_samples)
    mean, std = stats["mean_duty_cycle"], stats["std_duty_cycle"]
    problem_fraction = stats["problem_fraction"] or 0.0
    if mean is not None:
        classification = "spatially_varying" if problem_fraction > 0 else "ok"
    elif stats["n_domain_a_found"] <= 1:
        classification = "severely_overpoled"
    else:
        classification = "unmeasurable"
    diagnostics = {
        "problem_fraction": problem_fraction,
        "classification": classification,
        "n_normal": stats["n_normal"],
        "n_underpoled_region": stats["n_underpoled_region"],
        "n_domain_a_found": stats["n_domain_a_found"],
        "duty_cycle_stats": stats,
    }
    return mean, std, diagnostics


def draw_domain_boundary_lines(image_path, angle=None, upscale=4, x_spacing=3, n_samples=60):
    """Draws: the tip-to-tip connector for every detected tip row (orange solid through its own
    two independently-detected L/R points, or cyan dashed for an interpolated/made-up row), each
    fitted domain boundary line across the full ROI width (magenta solid) for validated domain-A
    instances only, its inlier points (green dots, at their real column) and outlier points (red
    X, at their real column) that got rejected, and one thick green vertical tick per domain-A
    instance showing its width (upper boundary to lower boundary) at a representative column. See
    compute_domain_boundaries() for the detection/validation logic this renders."""
    data = compute_domain_boundaries(image_path, angle=angle, n_samples=n_samples)
    deskewed, x_left, x_right = data["deskewed"], data["x_left"], data["x_right"]
    lo_pairs, hi_pairs = data["lo_pairs"], data["hi_pairs"]
    tip_rows, made_up_rows, x_per_row = data["tip_rows"], data["made_up_rows"], data["x_per_row"]
    tip_fns = data["tip_fns"]
    upper_lines, lower_lines, period_indices = data["upper_lines"], data["lower_lines"], data["period_indices"]

    canvas = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    # tip-to-tip connector for every tip row -- orange solid through the two independently-detected
    # L/R metal-edge points for a real, detected tip; cyan dashed for an interpolated (made-up) one,
    # which has no real detected points and is just a flat stand-in at that row.
    for i, tip_row in enumerate(tip_rows):
        y_l = int(round(tip_fns[i](x_left)))
        y_r = int(round(tip_fns[i](x_right)))
        if tip_row in made_up_rows:
            for x in range(x_left, x_right, 6):
                y = int(round(tip_fns[i](x)))
                cv2.line(canvas, (x, y), (min(x + 3, x_right), y), (255, 255, 0), 1)
        else:
            cv2.line(canvas, (x_left, y_l), (x_right, y_r), (0, 140, 255), 1)

    # only draw a boundary's line/dots if it belongs to a validated pair -- a lone or eliminated
    # boundary isn't a real domain-A edge, so it shouldn't appear as if it were one.
    for i in period_indices:
        for line in (upper_lines[i], lower_lines[i]):
            for x0, r in line["inliers"]:
                cv2.circle(canvas, (x0, int(round(r))), 1, (80, 220, 80), -1)
            for x0, r in line["outliers"]:
                cv2.drawMarker(canvas, (x0, int(round(r))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 5, 1)
            y_l = int(round(_eval_line(line, x_left)))
            y_r = int(round(_eval_line(line, x_right)))
            cv2.line(canvas, (x_left, y_l), (x_right, y_r), (255, 0, 255), 1)
    # spread ticks evenly WITHIN [x_left, x_right] -- the requested x_spacing is a maximum, not a
    # fixed step: with many periods (this dataset can have 40+ in one window), x_start + n*x_spacing
    # can run past x_right, drawing ticks outside the ROI entirely.
    n_ticks = max(1, len(period_indices))
    inner_width = max(1, (x_right - x_left) - 2 * x_spacing)
    step = min(x_spacing, inner_width / n_ticks) if n_ticks > 1 else 0
    x_tick_start = x_left + x_spacing
    for j, i in enumerate(period_indices):
        row_tip = tip_rows[i]
        x_anchor = int(round(x_tick_start + j * step))
        upper_row = _eval_line(upper_lines[i], x_anchor)
        lower_row = _eval_line(lower_lines[i], x_anchor)
        # tick follows the same small tilt as everything else (x_per_row, the metal boundary's
        # own measured slant) instead of being drawn as a forced-vertical line -- per the user's
        # correction, it should read as parallel to the tilted ROI boundary lines, not upright.
        x_upper = int(round(x_anchor + (row_tip - upper_row) * x_per_row))
        x_lower = int(round(x_anchor + (row_tip - lower_row) * x_per_row))
        cv2.line(canvas, (x_upper, int(round(upper_row))), (x_anchor, row_tip), (0, 255, 0), 2)
        cv2.line(canvas, (x_anchor, row_tip), (x_lower, int(round(lower_row))), (0, 255, 0), 2)

    y0, y1 = max(0, lo_pairs - 10), min(deskewed.shape[0], hi_pairs + 10)
    crop = canvas[y0:y1]
    crop = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{stem}_boundary_lines.png")
    cv2.imwrite(out_path, crop)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tip_lines_check.py <image_path> [angle_deg]")
        sys.exit(1)
    path = sys.argv[1]
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else None
    draw_tip_lines(path, angle)
    draw_domain_widths(path, angle)
