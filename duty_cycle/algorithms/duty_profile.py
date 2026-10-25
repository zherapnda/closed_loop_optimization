import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks, peak_prominences
from scipy import ndimage

from loader import to_gray
from edge import detect_boundary_tilt, detect_edge_points


DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image2.png")

# manual angles saved via duty_cycle/deskew_check.py -- automatic detection is
# unreliable for a meaningful fraction of images (confirmed: gets pulled onto
# the wrong finger near the ROI edges even after several fixes), so a manually
# verified angle takes priority over the automatic guess whenever one exists
# for that exact image path.
MANUAL_ANGLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manual_angles.json")


def get_manual_angle(image):
    """image may be a path (str) or an already-loaded array -- manual angles
    are only ever looked up by path, since an in-memory array has no path to
    match against. Returns None if no path, no saved file, or no match."""
    if not isinstance(image, str) or not os.path.exists(MANUAL_ANGLES_PATH):
        return None
    import json
    with open(MANUAL_ANGLES_PATH) as f:
        angles = json.load(f)
    return angles.get(os.path.abspath(image))
ROI_HEIGHT = 400  # rows scanned upward from the found bottom line

MIN_PEAK_DISTANCE = 2
PROMINENCE = 2
WLEN = 21  # samples -- bounds prominence/width search so it can't reach past neighboring peaks

# a heavily overpoled sample can grow domain A until it blends into domain B,
# erasing the trough between them below the prominence threshold. find_peaks
# then reports one merged domain roughly 2x normal size instead of two. Any
# domain length beyond this fraction of the column's own median flags a likely
# missed trough. Calibrated empirically, not guessed: on a known-clean
# reference image, ordinary noise pushed this ratio as high as 2.65x (99th
# percentile across 148 columns) with no real merges present. A genuinely
# overpoled sample showed ratios of 3.1x-5.5x for its merged domains. 3.0
# sits cleanly between the two, so it catches real merges without flagging
# normal variation.
MERGE_RATIO_THRESHOLD = 3.0

# WLEN and MIN_PEAK_DISTANCE above only make sense at the pixel scale they were
# tuned on (~9-10px domain lengths). A different image resolution/magnification
# of the same physical structure can have a completely different domain length
# in pixels, so both are re-derived per image from an autocorrelation-estimated
# domain length instead of used as fixed constants. These ratios are calibrated
# against the original tuning: WLEN=21 for a ~9-10px domain length -> ~2.2x;
# MIN_PEAK_DISTANCE=2 for the same -> ~0.2x.
WLEN_TO_DOMAIN_RATIO = 2.2
MIN_DISTANCE_TO_DOMAIN_RATIO = 0.2

# a column is dropped from the ROI if its ridge contrast (median peak prominence)
# falls below this fraction of the best contrast seen anywhere in the sweep --
# catches shadow-degraded columns even though the degradation is a smooth
# gradient with no single boundary to detect
CONTRAST_FRACTION = 0.3

# minimum fraction of a "problem" segment's columns that must measure
# confidently before that segment's valid columns count toward the overall
# mean/std -- keeps a mostly-collapsed segment (a few lucky columns out of
# many) from sneaking into the average, while still crediting a segment
# where a real, substantial chunk of it measured fine (see the note where
# this is used, below the per-segment loop, for the concrete case that
# motivated it).
SEGMENT_INCLUDE_MIN_VALID_FRACTION = 0.2

# a mildly overpoled/underpoled region doesn't always fail outright (find_peaks
# still finds troughs and computes a ratio) -- it can instead swing wildly
# column to column (domains are irregular rather than fully undetectable),
# averaging out to a mean deceptively close to 50% even though the region is
# genuinely not uniform. Rolling-window std catches this instability directly.
# Calibrated empirically: a known-clean reference image's rolling std (window
# of 15 columns) never exceeded 1.81% (99th percentile across 148 columns). A
# region confirmed overpoled by hand-annotation showed ~9%. 3.5 sits well
# above the clean ceiling and well below the confirmed-unstable region.
UNSTABLE_STD_THRESHOLD = 3.5
UNSTABLE_STD_WINDOW = 15

# duty cycle = domain_A / (domain_A + domain_B), so its sign relative to 50%
# *is* the definition of over- vs under-poled -- this doesn't need the kind of
# empirical calibration MERGE_RATIO_THRESHOLD/UNSTABLE_STD_THRESHOLD above
# required, since it's reading the physical quantity directly rather than a
# proxy for it. The tolerance just keeps ordinary measurement noise around
# 50% from being labeled a direction at all.
TARGET_DUTY_CYCLE = 50.0  # percent
DIRECTION_TOLERANCE = 2.0  # percentage points around TARGET_DUTY_CYCLE that count as "balanced"
BIDIRECTIONAL_FRACTION = 0.15  # a region needs at least this fraction of columns on EACH side to call it bidirectional


def classify_direction(dcs, tolerance=DIRECTION_TOLERANCE, bidirectional_fraction=BIDIRECTIONAL_FRACTION):
    """Read over/under-poled direction straight off the measured per-column
    duty cycles (percent), not off a failure-mode proxy. Only meaningful for
    columns that produced a real ratio -- a hard failure (no trough survived
    anywhere) has no ratio to read a sign from, so this returns None for an
    empty input rather than guessing; callers should keep those labeled by
    their failure signature instead (see "severely_overpoled" in
    compute_duty_cycle).

    Checked for bidirectional spread before falling back to the mean: an
    "unstable" region can genuinely swing both above and below target rather
    than sitting on one side with noise, and averaging that away would hide
    exactly the kind of non-uniformity this whole segmentation scheme exists
    to surface."""
    dcs = np.asarray([d for d in dcs if d is not None], dtype=float)
    if len(dcs) == 0:
        return None
    frac_over = float(np.mean(dcs > TARGET_DUTY_CYCLE + tolerance))
    frac_under = float(np.mean(dcs < TARGET_DUTY_CYCLE - tolerance))
    if frac_over >= bidirectional_fraction and frac_under >= bidirectional_fraction:
        return "bidirectional"
    diff = float(np.mean(dcs)) - TARGET_DUTY_CYCLE
    if diff > tolerance:
        return "overpoled"
    if diff < -tolerance:
        return "underpoled"
    return "balanced"


def classification_direction_key(classification):
    """Bucket a segment's fine-grained classification string (e.g.
    "unstable_likely_underpoled") down to its coarse direction, shared by the
    top-level summary rollup and the visualization coloring so they can't
    drift out of sync with each other."""
    if classification == "ok":
        return "ok"
    if classification == "severely_overpoled":
        return "severely_overpoled"
    if "bidirectional" in classification:
        return "bidirectional"
    if "underpoled" in classification:
        return "underpoled"
    if "overpoled" in classification:
        return "overpoled"
    return "indeterminate"


SEGMENT_COLOR_BGR = {
    "ok": (0, 255, 255),
    "overpoled": (0, 0, 255),
    "underpoled": (255, 0, 0),
    "bidirectional": (255, 0, 255),
    "severely_overpoled": (0, 0, 255),
    "indeterminate": (0, 140, 255),
}
SEGMENT_COLOR_MPL = {
    "overpoled": "red",
    "underpoled": "blue",
    "bidirectional": "magenta",
    "severely_overpoled": "red",
    "indeterminate": "orange",
}

BAND_HEIGHT = 8  # pixels -- thick enough to average out noise, thin enough to stay within one domain
# a candidate band's row-to-row roughness (std of the horizontal derivative,
# normalized by mean brightness) must stay under this fraction to count as
# "inside a domain, not crossing a finger edge". Roughness (not raw min/max
# range) because the ROI has a real left-right illumination gradient that
# would otherwise swamp the range even for a genuinely flat, in-domain row.
# Recalibrated after switching the deskew method: the smaller/noisier poling
# images' best achievable roughness (excluding the border margin below) ran
# 0.017-0.023, versus ~0.037+ for rows that actually cross a finger edge --
# 0.025 sits between the two, same as the old 0.006 did for the cleaner,
# larger reference images.
ROUGHNESS_FRACTION = 0.025
# rows within this fraction of the top/bottom are excluded from the search --
# they can be trivially "flat" (a vignette/border area with no real domain
# pattern) rather than a genuine mid-domain row, which would otherwise win as
# a false "best" candidate. Matches EDGE_MARGIN_FRACTION's reasoning below.
BOTTOM_LINE_MARGIN_FRACTION = 0.15


DESKEW_SKIP_TROUGHS = 2  # skip the first few troughs from the top when picking a start point -- edge/vignette artifacts live there
DESKEW_SEARCH_HALF_WINDOW = 3  # px each side -- how far a tracked trough can move column to column


def track_finger_points(gray, x_left, x_right, search_height=ROI_HEIGHT,
                         skip_troughs=DESKEW_SKIP_TROUGHS, search_half_window=DESKEW_SEARCH_HALF_WINDOW):
    """Track a single dark trough (one metal finger) column by column across
    x_left:x_right, following the nearest local minimum from one column to
    the next within a small window. Returns the list of (x, y) points tracked
    -- exposed separately from find_deskew_angle so a caller (e.g. a visual
    verification tool) can see exactly where the tracker went, not just the
    final fitted angle.

    The starting trough is picked a few troughs in from the top (not the
    first one), since the very first can be a vignette/edge artifact rather
    than a real finger."""
    height = min(gray.shape[0], search_height)

    # the very first column can occasionally lack enough troughs to start
    # from (e.g. it happens to sit on a low-contrast stretch) -- try nearby
    # columns instead of failing the whole image over one unlucky column
    start_x = None
    y_start = None
    for candidate_x in range(x_left, min(x_left + 10, x_right)):
        first_column = gray[0:height, candidate_x].astype(float)
        troughs, _ = find_peaks(-first_column, prominence=PROMINENCE, distance=1)
        if len(troughs) > skip_troughs:
            start_x, y_start = candidate_x, int(troughs[skip_troughs])
            break
    if start_x is None:
        raise RuntimeError("Not enough troughs detected near x_left to start tracking a finger line.")

    points = [(start_x, y_start)]
    y_current = y_start
    for x in range(start_x + 1, x_right):
        lo = max(0, y_current - search_half_window)
        hi = min(height, y_current + search_half_window + 1)
        segment = gray[lo:hi, x].astype(float)
        if len(segment) == 0:
            continue
        y_found = lo + int(np.argmin(segment))
        points.append((x, y_found))
        y_current = y_found

    return points


def angle_from_points(points):
    """The rotation angle to pass to deskew() so the line through these
    (x, y) points becomes horizontal. Validated against a synthetic image
    with a known ground-truth tilt (a controlled test, not a proxy metric):
    normalize fitLine's direction to point in the increasing-x direction
    (its raw sign is documented as arbitrary and was confirmed NOT reliably
    tied to input order), then use the angle directly -- no extra sign flip.
    This is the convention for clean point pairs (e.g. manually clicked in
    deskew_check.py); find_deskew_angle's automatic tracker uses its own
    separately-validated sign, since its tracked points behave differently
    (see the comment there)."""
    pts_arr = np.array(points, dtype=np.float32)
    vx, vy, _, _ = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vy, vx)))


def find_deskew_angle(gray, x_left, x_right, search_height=ROI_HEIGHT,
                       skip_troughs=DESKEW_SKIP_TROUGHS, search_half_window=DESKEW_SEARCH_HALF_WINDOW):
    """Find the grating tilt by tracking a single dark trough (one metal
    finger) column by column across x_left:x_right, then fitting a robust
    line through the tracked points -- the same line-of-best-fit approach
    already used for the boundary edges (edge.py), applied to one periodic
    dark line instead of the outer boundary. This is not the same angle as
    detect_roi's boundary-edge slope, which only straightens the outer edges
    of the whole comb structure and leaves the individual finger lines still
    tilted.

    Known unreliable on some real images (confirmed via manual click
    comparison and a failed consensus-of-many-tracks experiment: individual
    tracks starting from different troughs on the same image often disagree
    wildly with each other, so voting across them doesn't help -- the
    per-column greedy tracking itself is the weak point, not this specific
    starting trough). Kept as the best-known automatic fallback for images
    without a manual override; get_manual_angle() should be checked first
    wherever precision matters (see compute_duty_cycle).

    Empirically, this automatic tracker's points need the OPPOSITE sign from
    angle_from_points' now-validated convention (confirmed via direct visual
    check on real data) -- the column-by-column tracking process apparently
    produces a systematically different effective direction than a clean
    point pair does. Negated explicitly here rather than baking that into
    angle_from_points, which is also used for manually-clicked points that
    don't have this quirk."""
    points = track_finger_points(gray, x_left, x_right, search_height, skip_troughs, search_half_window)
    return -angle_from_points(points)


FFT_PAD_FACTOR = 4  # zero-pad this much before the FFT for sub-pixel frequency resolution
FFT_BAND_FRAC = 0.4  # search peaks within +/-40% of the expected fundamental frequency
FFT_DC_EXCLUDE_RADIUS = 3  # px (pre-padding) around DC to always exclude, regardless of expected period


def find_deskew_angle_fft(gray, x_left, x_right, search_height=ROI_HEIGHT,
                           pad_factor=FFT_PAD_FACTOR, band_frac=FFT_BAND_FRAC):
    """Estimate the grating tilt directly from the periodic pattern's
    frequency-domain signature, instead of tracking any single point down the
    image column by column. track_finger_points' greedy per-column search
    occasionally locks onto the wrong local minimum and drifts onto an
    adjacent finger, which is what produces find_deskew_angle's wildly wrong
    angles on some images -- the FFT has no "previous column's answer" to
    drift from, since the whole patch votes on one dominant orientation at
    once.

    A naive global-max peak search over the full spectrum instead picks up
    two kinds of noise: the DC-adjacent smear from the smooth left-right
    brightness gradient across the ROI, and -- even once that's excluded --
    windowing/edge artifacts that can outweigh the real finger-spacing peak
    (confirmed by inspecting the raw spectrum: without restriction, the
    top peaks cluster right next to DC, not near the pattern's actual
    frequency). Since the approximate period is already knowable from a
    single column's trough spacing (estimate_domain_length), the search is
    restricted to an annulus of frequency radii around that expected
    fundamental, which reliably isolates the true periodic peak from both
    kinds of spurious energy."""
    height = min(gray.shape[0], search_height)
    patch = gray[0:height, x_left:x_right].astype(float)

    mid_col = patch[:, patch.shape[1] // 2]
    period = 2 * estimate_domain_length(mid_col)

    patch = patch - patch.mean()
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    patch = patch * window

    padded_shape = (patch.shape[0] * pad_factor, patch.shape[1] * pad_factor)
    F = np.fft.fftshift(np.fft.fft2(patch, s=padded_shape))
    mag = np.abs(F)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2

    expected_radius = padded_shape[0] / period
    lo_r = max(expected_radius * (1 - band_frac), FFT_DC_EXCLUDE_RADIUS * pad_factor)
    hi_r = expected_radius * (1 + band_frac)

    yy, xx = np.mgrid[0:mag.shape[0], 0:mag.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus = (dist >= lo_r) & (dist <= hi_r) & (yy > cy)
    mag_masked = np.where(annulus, mag, 0.0)
    if not np.any(mag_masked):
        raise RuntimeError(
            "No frequency energy found in the expected periodicity band -- "
            "x_left:x_right may not contain a clean periodic pattern.")

    py, px = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)

    # sub-pixel peak refinement via parabolic interpolation along each axis
    px_f, py_f = float(px), float(py)
    if 0 < px < mag_masked.shape[1] - 1:
        left, center, right = mag_masked[py, px - 1], mag_masked[py, px], mag_masked[py, px + 1]
        denom = left - 2 * center + right
        if denom != 0:
            px_f = px + 0.5 * (left - right) / denom
    if 0 < py < mag_masked.shape[0] - 1:
        above, center, below = mag_masked[py - 1, px], mag_masked[py, px], mag_masked[py + 1, px]
        denom = above - 2 * center + below
        if denom != 0:
            py_f = py + 0.5 * (above - below) / denom

    fy, fx = py_f - cy, px_f - cx
    return -float(np.degrees(np.arctan2(fx, fy)))


def deskew(gray, angle_deg):
    h, w = gray.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h))


ANCHOR_ROW_FRACTION = 0.5  # search for the stable band starting at the image center -- confirmed by direct
# visual inspection that the center of these images is clearer/more reliable than the upper area
MEASUREMENT_HEIGHT = 700  # rows scanned upward from the found bottom line for the actual duty-cycle
# measurement -- deliberately smaller than ROI_HEIGHT (which stays as the deskew-tracking search cap) so the
# measurement window sits centered around the found band instead of always stretching up to row 0.
# Was 100, tuned for ~6px domain periods (~16 periods per window). On the higher-resolution ZEHRA01
# images (~41px periods), 100 rows only fits 2-3 periods -- far too few for a robust confidence-sweep
# measurement. Superseded by MEASUREMENT_MARGIN_FRACTION below for that dataset (kept here as the
# fallback fixed height if a caller doesn't use the fraction-based window).

MEASUREMENT_MARGIN_FRACTION = 0.05  # use nearly the whole image height for measurement, excluding only
# this fraction as a margin at the top and bottom -- for the ZEHRA01 images specifically, which are tall
# enough (2048px) to contain dozens of periods, using a small fixed-height window like MEASUREMENT_HEIGHT
# wastes most of the available data. The excluded margin still needs to be wide enough to clear a full
# tooth so a partially cut-off one at the window edge doesn't get detected as a spurious extra tip (see
# TIP_SEARCH_MARGIN/TIP_PEAK_DISTANCE, scaled separately for the same reason).


def measurement_window(image_height, margin_fraction=MEASUREMENT_MARGIN_FRACTION):
    """(top_row, bottom_search_row) spanning nearly the full image height, excluding
    margin_fraction at each end. bottom_search_row is where find_tip_alignment_row should
    anchor its search (near the bottom of this window, not the image center) so bottom_row
    itself lands inside the window instead of the window being built around a center anchor."""
    margin = int(round(margin_fraction * image_height))
    return margin, image_height - margin


def find_domain_bottom_line(deskewed, x_left, x_right, band_height=BAND_HEIGHT, roughness_fraction=ROUGHNESS_FRACTION,
                             anchor_row_fraction=ANCHOR_ROW_FRACTION, margin_fraction=BOTTOM_LINE_MARGIN_FRACTION):
    """Search outward from a row near the top of the deskewed image (rather
    than dead center, so the ROI sits in the upper part of the image) for a
    horizontal band that's smooth across x_left:x_right -- i.e. a band that
    sits inside a single domain instead of straddling a finger edge. Its bottom
    row anchors the ROI. Since that row sits mid-domain-A, counting domains
    upward from there always starts partway through A: the first full
    trough-to-trough span is domain B, the next is domain A, alternating from
    there -- no brightness/width guess needed to tell the two apart.

    The top/bottom margin is excluded from the search entirely: a
    vignette/border area can be trivially flatter than anywhere in the real
    domain pattern (there's no pattern there to disrupt the flatness), which
    would otherwise win as a false "best" candidate and anchor the ROI outside
    the actual grating."""
    height = deskewed.shape[0]
    margin = int(height * margin_fraction)
    anchor = int(height * anchor_row_fraction)
    offsets = [0]
    for d in range(1, max(anchor, height - anchor)):
        offsets += [d, -d]

    best_start, best_roughness = None, np.inf

    for offset in offsets:
        start = anchor + offset
        if start < margin or start + band_height > height - margin:
            continue
        profile = deskewed[start:start + band_height, x_left:x_right].astype(float).mean(axis=0)
        roughness = np.std(np.diff(profile)) / profile.mean()
        if roughness <= roughness_fraction:
            return start + band_height - 1
        if roughness < best_roughness:
            best_start, best_roughness = start, roughness

    if best_start is None:
        raise RuntimeError("No valid band found -- x_left:x_right may be out of bounds for this image.")

    # nothing cleared the threshold outright -- rather than fail the whole
    # image, fall back to the smoothest band actually found. Real images vary
    # in contrast/lighting enough that a fixed threshold won't always clear;
    # a slightly-off anchor is far less costly than losing the image entirely.
    print(f"Warning: no band met roughness <= {roughness_fraction}; using best found ({best_roughness:.4f}) at row {best_start}")
    return best_start + band_height - 1


TIP_SEARCH_MARGIN = 50  # rows -- exclude near top/bottom edges from tip search (vignette/border artifacts).
# Was 20, tuned for ~6px domain periods. On the ZEHRA01 high-resolution images (~41px periods), a 20px
# margin doesn't even clear half a tooth, so scaled to comfortably clear a full period instead of an old
# absolute pixel count. (A near-duplicate detection 3px apart was seen on real data well clear of any
# window edge, though -- that turned out to be TIP_PEAK_DISTANCE below, not this margin.)
TIP_ALIGNMENT_TOLERANCE = 6  # rows -- how close independent left/right tip peaks must be to count as "aligned".
# Was 2, tuned for ~6px domain periods. On a higher-resolution image (~41px periods), the same small
# residual tilt that used to shift a matched pair by 1-2px shifts it by up to 5px instead -- confirmed
# directly (max observed diff among 14 genuinely-matching pairs was 5px) -- so the absolute tolerance
# needs to scale with image resolution, not stay fixed in pixels.
TIP_PEAK_DISTANCE = 20  # rows -- min spacing between tip peaks in the per-row boundary-position search.
# Was 3, tuned for ~6px domain periods. On the ZEHRA01 high-resolution images (~41px periods), a 3px
# minimum spacing does nothing to stop ordinary per-row detection noise within a SINGLE tooth from
# registering as two separate peaks -- confirmed on real data (G7): a spurious "detection" only 3px
# from a real one, well clear of any window edge, went away once this scaled with the new period size.
TIP_PEAK_PROMINENCE = 2  # px -- min prominence for a row's boundary position to count as a real tip, not noise
TIP_MIN_DENSITY_FRACTION = 0.5  # expect at least this fraction of the WEAKER side's own peak count to
# find an aligned partner. Relative to peaks actually found, not a fixed row-count divisor, so it scales
# with the image's own period size instead of assuming ~6px domains -- a fixed divisor tuned for 6px
# periods (needing e.g. 40 aligned pairs in a 600-row window) can never be satisfied on an image with
# ~41px periods, where only ~15 periods physically fit in that window regardless of alignment quality.
# A sparser-than-this intersection means requiring both sides to agree is throwing away good data, not
# filtering out bad data -- see the fallback below.


def _find_tip_peak_pairs(deskewed, lo, hi, tolerance=TIP_ALIGNMENT_TOLERANCE):
    """Core detection, returning full per-side pairing info (not just a
    collapsed row number) so a caller that wants to draw or reason about the
    actual two finger tips being matched -- not an average -- has access to
    them. See _find_tip_peak_rows for the simplified row-only view used by
    the production measurement path.

    Requiring both sides to agree assumes both sides are equally reliable,
    which isn't always true: confirmed on real images that one boundary side
    can be far too low-contrast for reliable peak detection (H8's left side
    found ZERO peaks at all despite a clean right side; D11's left side
    oscillates at near-noise amplitude, std=1.45px, vs its right side's
    std=11px). When that happens, requiring agreement doesn't filter out bad
    data -- it throws away all the good data from the reliable side, leaving
    too few (or zero) periods to measure anything. So the intersection is
    only trusted when it's dense enough to plausibly reflect the real domain
    period; otherwise this falls back to whichever single side found more
    peaks, trusting it directly rather than demanding corroboration it can't
    get.

    Returns (mode, pairs):
      mode is "aligned", "left_only", or "right_only".
      In "aligned" mode, each pair is
        {"row": avg_row, "row_l": lr, "x_l": xl, "row_r": rr, "x_r": xr} --
        row_l/row_r are the ACTUAL rows each side's own tip peaked at
        (deliberately kept separate, not just their average: a deskew angle
        derived from one calibration slice doesn't perfectly align every
        finger pair across the whole strip, so the true tip-to-tip
        connector isn't guaranteed to be horizontal).
      In fallback mode, each pair is {"row": r, "x": x} from the single
      reliable side only -- there's no opposite-side point to pair it with,
      so no connector can be drawn for these."""
    all_rows = np.arange(lo, hi)
    left_points = detect_edge_points(deskewed, all_rows, from_right=False)
    right_points = detect_edge_points(deskewed, all_rows, from_right=True)
    left_by_row = {y: x for x, y in left_points}
    right_by_row = {y: x for x, y in right_points}

    rows_l = np.array(sorted(left_by_row.keys()))
    rows_r = np.array(sorted(right_by_row.keys()))
    if len(rows_l) < 3 or len(rows_r) < 3:
        raise RuntimeError("Not enough boundary points detected to find tip-alignment rows.")
    vals_l = np.array([left_by_row[r] for r in rows_l])
    vals_r = np.array([right_by_row[r] for r in rows_r])

    # left_by_row/right_by_row are both the ABSOLUTE x of the domain<->metal transition on
    # their respective side (detect_edge_points scans inward from each edge to find it). As a
    # LEFT tooth protrudes further into the domain, that transition moves to a LARGER x (a peak
    # in vals_l). As a RIGHT tooth protrudes further into the domain, the transition moves to a
    # SMALLER x instead (a trough in vals_r, not a peak) -- confirmed on a real high-resolution
    # image where using find_peaks on both sides identically put the right-side "tip" on the gap
    # between two teeth instead of on a tooth, visually obvious once the image was sharp enough
    # to tell the two apart (lower-resolution images likely had the same bug, just not visibly).
    peaks_l, _ = find_peaks(vals_l, distance=TIP_PEAK_DISTANCE, prominence=TIP_PEAK_PROMINENCE)
    peaks_r, _ = find_peaks(-vals_r, distance=TIP_PEAK_DISTANCE, prominence=TIP_PEAK_PROMINENCE)
    peak_rows_l = rows_l[peaks_l]
    peak_rows_r = rows_r[peaks_r]

    aligned = {}
    if len(peak_rows_l) and len(peak_rows_r):
        for lr in peak_rows_l:
            rr = peak_rows_r[np.argmin(np.abs(peak_rows_r - lr))]
            if abs(rr - lr) <= tolerance:
                avg_row = int(round((lr + rr) / 2))
                aligned[avg_row] = {
                    "row": avg_row,
                    "row_l": int(lr), "x_l": int(left_by_row[lr]),
                    "row_r": int(rr), "x_r": int(right_by_row[rr]),
                }

    expected_min = TIP_MIN_DENSITY_FRACTION * min(len(peak_rows_l), len(peak_rows_r))
    if len(peak_rows_l) and len(peak_rows_r) and len(aligned) >= expected_min:
        return "aligned", sorted(aligned.values(), key=lambda p: p["row"])

    if len(peak_rows_l) == 0 and len(peak_rows_r) == 0:
        raise RuntimeError("No finger-tip peaks detected on either side -- cannot anchor via tip alignment.")
    if len(peak_rows_l) >= len(peak_rows_r):
        mode, fallback, by_row = "left_only", peak_rows_l, left_by_row
    else:
        mode, fallback, by_row = "right_only", peak_rows_r, right_by_row
    pairs = [{"row": int(r), "x": int(by_row[r])} for r in sorted(fallback.tolist())]
    return mode, pairs


def _find_tip_peak_rows(deskewed, lo, hi, tolerance=TIP_ALIGNMENT_TOLERANCE):
    """Row-only view of _find_tip_peak_pairs, for callers (the production
    measurement path) that just need period-boundary anchors and don't need
    the individual per-side tip positions."""
    _, pairs = _find_tip_peak_pairs(deskewed, lo, hi, tolerance)
    return [p["row"] for p in pairs]


def find_tip_alignment_row(deskewed, search_center_row, half_window=100,
                            tolerance=TIP_ALIGNMENT_TOLERANCE, margin=TIP_SEARCH_MARGIN):
    """Find the row where the left and right metal finger tips are aligned --
    unlike find_domain_bottom_line's flatness search, this pins down domain
    identity unambiguously rather than assuming it. Flatness alone can't
    tell domain A from domain B (both are equally flat internally), and
    getting that assumption wrong silently inverts the whole image's duty
    cycle reading (over<->under swapped) -- confirmed as a real, recurring
    problem, not a hypothetical one.

    Domain A is physically defined as the gap between two opposing finger
    tips, so a row where the tips are aligned is, by definition, the point
    where that gap is narrowest -- i.e. the center of one instance of
    domain A (see find_all_tip_alignment_rows and duty_cycle_for_signal,
    which anchor period-by-period measurement to this same definition).

    Each side's boundary x-position oscillates row to row with the same
    period as the domain pattern (confirmed on a real image: independently
    detected left/right tip rows agreed to within 1-2 rows for 33/38 of
    the pairs found) -- individual finger tips show up as clean local
    maxima in that oscillation. Searches within
    [search_center_row-half_window, +half_window] and returns the aligned
    pair closest to search_center_row, so the caller can request an anchor
    near the image center without depending on exactly which period happens
    to land there."""
    height = deskewed.shape[0]
    lo = max(margin, search_center_row - half_window)
    hi = min(height - margin, search_center_row + half_window)
    aligned_rows = _find_tip_peak_rows(deskewed, lo, hi, tolerance)
    if not aligned_rows:
        raise RuntimeError("No aligned left/right finger-tip pair found -- cannot anchor via tip alignment.")
    return min(aligned_rows, key=lambda r: abs(r - search_center_row))


def find_all_tip_alignment_rows(deskewed, row_lo, row_hi, tolerance=TIP_ALIGNMENT_TOLERANCE, margin=TIP_SEARCH_MARGIN):
    """Find every tip-alignment row (not just the one closest to some target)
    within [row_lo, row_hi]. Each one marks the center of one physical
    instance of domain A, so the gap between two CONSECUTIVE aligned rows is
    the true period (domain A + domain B) at that point in the image,
    measured via the metal boundary's own high-contrast oscillation rather
    than the more fragile domain-boundary troughs. See
    duty_cycle_for_signal's period-based measurement, which uses this.

    Raw detection only -- doesn't correct for missed tips (see
    interpolate_missing_tips, which every caller of this should run the
    result through before treating consecutive rows as one real period)."""
    lo = max(margin, row_lo)
    hi = min(deskewed.shape[0] - margin, row_hi)
    return _find_tip_peak_rows(deskewed, lo, hi, tolerance)


GAP_OUTLIER_RATIO = 1.6  # a consecutive-row spacing more than this many times the image's own median
# spacing is treated as "not really consecutive" -- almost certainly 1+ missed tip detections in between,
# not one real (very wide) period. A real chip's period spacing is approximately constant, so the median
# across all detected gaps is a solid "normal" reference. 1.6x leaves room for ordinary jitter (a 6px
# period wobbling to 5-7px is a 1.4x spread at most) while still catching a clearly-merged gap. Visually
# confirmed on a real image (D14): a 2.0x gap had exactly 1 missed tip in between, and a 4.0x gap had
# exactly 3 -- i.e. round(gap/median)-1 matched the true miss count exactly, not just approximately.


def interpolate_missing_tips(rows, outlier_ratio=GAP_OUTLIER_RATIO):
    """Insert evenly-spaced synthetic tip rows into any gap between
    consecutive DETECTED tip-alignment rows that's a statistical outlier
    relative to the image's own typical spacing, instead of silently
    treating that whole gap as one real (very wide) period -- confirmed as
    a real, recurring failure mode: D14 had a 24px gap where the true
    spacing elsewhere was 6px, and duty_cycle_for_signal treated it as one
    period, feeding a badly wrong domain-A/period ratio into the average
    for that whole span.

    The inserted rows are a deliberate, labeled ESTIMATE (assumed evenly
    spaced at the image's own median period, per GAP_OUTLIER_RATIO), not a
    detection -- there's no way to know the missed tips' exact positions
    without seeing them independently confirmed. But since a real chip's
    period spacing is approximately constant (confirmed: the estimated miss
    count from round(gap/median)-1 matched the true miss count exactly on a
    real image, see GAP_OUTLIER_RATIO), assuming uniform spacing recovers
    real measurable periods instead of one corrupted giant one."""
    rows = sorted(rows)
    if len(rows) < 2:
        return rows
    gaps = np.diff(rows)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return rows

    result = [rows[0]]
    for i, gap in enumerate(gaps):
        if gap > outlier_ratio * median_gap:
            n_missed = max(0, int(round(gap / median_gap)) - 1)
            for k in range(1, n_missed + 1):
                result.append(int(round(rows[i] + k * gap / (n_missed + 1))))
        result.append(rows[i + 1])
    return result


def strip_signal(deskewed, x0, bottom_row, height=ROI_HEIGHT, x_per_row=0.0):
    """Extract a vertical (or, with x_per_row != 0, gently slanted) strip of
    pixel intensities running upward from bottom_row -- index 0 = bottom_row,
    increasing toward top_row. x_per_row shifts the sampled column by that
    many pixels per row of upward travel, letting the strip follow the metal
    boundary's own measured tilt (edge.detect_boundary_tilt, re-measured on
    the deskewed image) instead of assuming a perfectly vertical column is
    correct post-deskew -- not guaranteed unless the boundary's tilt and the
    deskew (domain-grating) angle happen to be identical."""
    top_row = max(0, bottom_row - height)
    n = bottom_row - top_row + 1
    rows = bottom_row - np.arange(n)
    width = deskewed.shape[1]
    xs = np.clip(np.round(x0 + np.arange(n) * x_per_row).astype(int), 0, width - 1)
    return deskewed[rows, xs].astype(float)


def estimate_domain_length(signal, min_period=4, max_period=150, bootstrap_wlen=15, bootstrap_prominence=2):
    """Estimate the spatial domain length (in pixels), so WLEN/MIN_PEAK_DISTANCE
    can be derived per image instead of assuming a fixed pixel scale tuned on a
    different image resolution/magnification.

    Primary method: a direct trough detection with a small, scale-agnostic
    window, taking the median gap. Autocorrelation (the original approach) is
    used only as a fallback for low-contrast signals where that direct pass
    finds too few troughs -- autocorrelation assumes clean self-similar
    periodicity, which breaks down for irregular domain lengths (e.g. an
    underpoled sample where domain lengths vary a lot column to column) and
    can silently lock onto a spurious multi-period pattern instead of the
    real, much shorter domain length."""
    troughs, _ = find_peaks(-signal, distance=1, prominence=bootstrap_prominence, wlen=bootstrap_wlen)
    if len(troughs) >= 3:
        return max(min_period, int(np.median(np.diff(troughs))))

    detrended = signal - signal.mean()
    autocorr = np.correlate(detrended, detrended, mode="full")
    autocorr = autocorr[autocorr.size // 2:]
    max_period = min(max_period, len(autocorr) - 1)
    search = autocorr[min_period:max_period + 1]
    if len(search) == 0:
        return min_period
    peaks, _ = find_peaks(search)
    if len(peaks) == 0:
        return min_period + int(np.argmax(search))
    return min_period + int(peaks[0])


def refine_trough_position(signal, i):
    """Sub-pixel-refine an integer trough index via parabolic interpolation
    over its immediate neighbors. At low pixel-per-domain resolution (a handful
    of px per domain), integer-only trough positions quantize the duty cycle
    ratio heavily -- e.g. every measurement near a 6px domain length rounds to
    one of a few coarse fractions, and the most common one (6/12 = 0.5) then
    dominates the median regardless of the real underlying value. This pulls
    sub-pixel information out of the intensity values around each trough
    instead of only trusting which integer sample was lowest."""
    if i <= 0 or i >= len(signal) - 1:
        return float(i)
    y0, y1, y2 = signal[i - 1], signal[i], signal[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    offset = 0.5 * (y0 - y2) / denom
    offset = max(-0.5, min(0.5, offset))  # clamp -- a noisy/flat neighborhood shouldn't extrapolate wildly
    return i + offset


# A fixed absolute PROMINENCE works fine for a high-contrast column but is too
# strict for a low-contrast one -- confirmed on a real image (E8, x=120): real,
# evenly-spaced troughs were there all along, just weak enough (prominence as
# low as 0.5) to be silently dropped at the default threshold, leaving only a
# few strong unrelated features that look like a totally different signal
# shape. Scaling prominence to the column's own IQR (interquartile range)
# adapts to that automatically. But a single adaptive threshold isn't enough
# either: sweeping a few reasonable fractions on that same column gave 57.8% /
# 55.2% / 39.2% -- genuinely unstable, not just noisy, meaning the exact
# trough positions are themselves ambiguous at that contrast level. A clean
# column (x=150) stayed at 49.5% across the same sweep. So instead of picking
# one threshold and hoping it's the right one, agreement ACROSS the sweep is
# itself the confidence check: only trust a column when multiple reasonable
# thresholds land on (roughly) the same answer.
CONFIDENCE_THRESHOLD_FRACTIONS = (0.3, 0.5, 0.7)  # fractions of each period's own baseline-to-trough
# depth, used as the half-min-style crossing level for locating domain B's valley extent (see
# duty_cycle_for_signal) -- swept the same way CONFIDENCE_PROMINENCE_FRACTIONS used to sweep prominence,
# so a period whose valley shape is ambiguous (result depends heavily on exactly where you draw the
# crossing line) still gets flagged as low-confidence rather than silently picking one answer.
CONFIDENCE_AGREEMENT_TOLERANCE = 25.0  # percentage points -- max spread allowed across the sweep.
# Originally tuned to 10.0 for the old prominence-sweep/single-trough model (5.0 flagged too much
# on clean reference images, 10.0 balanced it). Re-tuned to 25.0 after switching _valley_edges to
# measure against each period's own true peak/trough instead of its tip-row endpoints (see
# _valley_edges) -- that change makes the width measurement genuinely, not just noisily, more
# sensitive to exactly where the threshold_fractions sweep draws its crossing line: on a period
# with a gradual (not sharp) transition, CONFIDENCE_THRESHOLD_FRACTIONS 0.3/0.5/0.7 can legitimately
# disagree by 20+ points, since "where domain A ends" is inherently less sharply defined for a wide
# transition than a narrow one. At 10.0 this new method flagged 100% of columns low-confidence on
# every test image, D14/D11/H8 included -- not evidence of bad data, just a stale threshold from a
# different model. 25.0 restores reasonable coverage (matches what 10.0 gave under the old model)
# without materially changing the resulting median (confirmed stable to within ~1pp across 20-30).

MIN_RESOLVABLE_DOMAIN_PX = 1.0  # smallest domain width treated as distinguishable from "collapsed to
# nothing" -- used when a period's dip is too shallow to represent a real domain-B valley (see
# MIN_DIP_DEPTH_FRACTION below), not corrupted/unusable data. The true width below this floor genuinely
# can't be measured (that's what "below the image's resolution" means), so this is a deliberate, labeled
# estimate of the most extreme case that's still distinguishable from zero, not a claimed precise
# measurement.

MIN_DIP_DEPTH_FRACTION = 0.1  # fraction of the whole strip's own IQR -- a period whose brightness dip is
# shallower than this is indistinguishable from noise, so domain B is treated as fully collapsed
# (MIN_RESOLVABLE_DOMAIN_PX) rather than measured from a dip that isn't really there.


def _valley_edges(local_signal, frac, min_dip_depth):
    """Sub-pixel (left, right) position of the sub-threshold valley within
    one period, at the half-min-style crossing level `true_peak -
    frac*depth`, where true_peak/true_trough are the period's OWN actual
    brightness extremes (not the values at the period's tip-row endpoints).
    Sub-pixel-interpolates both edges of the crossing (same idea as
    refine_trough_position, applied to the valley's edges instead of its
    single deepest point) rather than snapping to the nearest integer
    sample.

    Treats the period as CIRCULAR: the valley is found by expanding
    outward from the period's true minimum in both directions until
    crossing back above threshold, rather than assuming it's a single
    contiguous span strictly between the two tip-row endpoints. This
    matters because a tip-alignment row marks a specific point in the
    metal boundary's OWN oscillation (where left/right tips are closest),
    which isn't guaranteed to land on domain A's brightness peak --
    confirmed on real data (700-1-1-15): every tip row there landed right
    next to the brightness TROUGH instead (e.g. endpoint value 43 with the
    period's true peak at 159), so the old "baseline = endpoint value"
    approach measured ~0 depth and treated nearly every period as fully
    collapsed, forcing a ~92% reading on a sample independently confirmed
    (by the user, cross-checked with their mentor) to be close to 50/50.
    With this period, the true minimum sits near the boundary between two
    tip rows, so the valley legitimately wraps around past index 0 or past
    the end of local_signal -- returned left/right can be <0 or >period
    accordingly; duty_cycle_for_signal only needs their difference, which
    is unaffected by the wrap, but a caller drawing this needs to clip to
    the period's own bounds.

    Confirmed this generalization doesn't just relocate the bug: it was
    tested against images where the OLD endpoint-based approach worked
    (D14, D11, F11 -- tip rows landing near the true peak there) and still
    gives sensible, correctly-directioned (overpoled) results, just at a
    somewhat lower magnitude than the old method's, because a period's true
    peak is always >= its endpoint average, giving a uniformly less
    generous "domain A" reference than before.

    Returns None when the dip is too shallow to represent a real valley
    (see MIN_DIP_DEPTH_FRACTION) -- there's no meaningful edge to report,
    only a floor-width estimate (see _valley_width, the caller most code
    should use)."""
    n = len(local_signal) - 1  # period length -- local_signal has period+1 samples (both endpoints)
    if n < 2:
        return None
    core = np.asarray(local_signal[:n], dtype=float)  # drop the duplicate final sample (same phase as index 0)
    true_peak = float(core.max())
    true_trough = float(core.min())
    depth = true_peak - true_trough
    if depth < min_dip_depth:
        return None

    threshold = true_peak - frac * depth
    argmin = int(core.argmin())
    rot = np.roll(core, -argmin)  # rot[0] = true minimum -- makes outward expansion wrap-safe

    r = 0
    while r + 1 < n and rot[r + 1] <= threshold:
        r += 1
    l = 0
    while l + 1 < n and rot[(n - 1 - l) % n] <= threshold:
        l += 1

    if r + 1 < n and rot[r] != rot[r + 1]:
        t = (rot[r] - threshold) / (rot[r] - rot[r + 1])
        right_ext = r + t
    else:
        right_ext = float(r)

    below_idx, above_idx = (n - l) % n, (n - 1 - l) % n
    if rot[below_idx] != rot[above_idx]:
        t = (rot[below_idx] - threshold) / (rot[below_idx] - rot[above_idx])
        left_ext = l + t
    else:
        left_ext = float(l)

    return argmin - left_ext, argmin + right_ext


def _valley_width(local_signal, frac, min_dip_depth):
    """Width of the sub-threshold valley found by _valley_edges, floored at
    MIN_RESOLVABLE_DOMAIN_PX -- used when no real valley is detected at all
    (collapsed dip, or a genuinely zero-width crossing), since a width
    below this floor isn't distinguishable from zero at this image's
    resolution (see MIN_RESOLVABLE_DOMAIN_PX)."""
    edges = _valley_edges(local_signal, frac, min_dip_depth)
    if edges is None:
        return MIN_RESOLVABLE_DOMAIN_PX
    left, right = edges
    return max(MIN_RESOLVABLE_DOMAIN_PX, right - left)


def duty_cycle_for_signal(signal, aligned_local_indices, verified_local_indices=None,
                           threshold_fractions=CONFIDENCE_THRESHOLD_FRACTIONS,
                           agreement_tolerance=CONFIDENCE_AGREEMENT_TOLERANCE):
    """Duty cycle per period = domain A / (domain A + domain B), measured
    period-by-period using tip-alignment rows (find_all_tip_alignment_rows,
    already converted to this signal's local index space by the caller) as
    period boundaries -- NOT by counting troughs sequentially down the whole
    strip. Each tip-alignment row marks the center of one instance of
    domain A (domain A is physically defined as the gap between two
    opposing finger tips), so the gap between two CONSECUTIVE tip-alignment
    rows is the true period (A+B), measured via the metal boundary's own
    high-contrast oscillation, not the more fragile domain-boundary troughs.

    Domain B's width within that period is measured as a VALLEY, not a
    single trough point: baseline = the average signal value at the
    period's two tip-row endpoints (both domain-A-like), and B's extent is
    where the signal dips below the midpoint between that baseline and the
    period's own darkest point (a half-min-style crossing). This replaced
    an earlier "double the distance from the tip row to the nearest single
    trough, assuming symmetry" model that looked right on a spot check but
    was confirmed broken on real data: whenever the one resolvable trough
    sat past the period's midpoint (common -- there's usually only one
    resolvable dip per period, not two separate A/B edges, because domain B
    is often narrow), doubling overshot the period entirely and silently
    hit the clamp below, producing the exact same capped ratio
    (period-1)/period regardless of the period's true shape. That artifact
    was traced directly: many genuinely different images were landing on
    identically 83.33% (5/6), which is the doubling-and-clamp's degenerate
    output for the dataset's common ~6px period, not a real measurement.
    The valley model measures each side of B's dip independently instead of
    assuming symmetry around the tip row, so it isn't fooled by an
    off-center dip the same way.

    If a period's dip is too shallow to represent a real valley at all
    (MIN_DIP_DEPTH_FRACTION), domain B has most likely compressed below
    this image's resolution -- treated as MIN_RESOLVABLE_DOMAIN_PX, not
    excluded as unusable data (confirmed on a real image, D14: excluding
    these regions as "unusable" left only the ordinary-looking columns to
    average, landing near 50% even though the region is severely overpoled;
    treating them as collapsed-B recovers ~75-80%, matching independently
    confirmed ground truth).

    This measures every period independently, anchored to a physically
    verified reference each time, instead of one global count-and-pair sweep
    where a single missed trough corrupts every pair after it -- the root
    cause of several earlier, confirmed bugs (parity corruption, ambiguous
    direction on a collapsed domain) in the previous sequential-counting
    approach.

    verified_local_indices (optional): the subset of aligned_local_indices
    whose row was an actually-DETECTED tip peak, as opposed to one inserted
    by interpolate_missing_tips to bridge an outlier gap. When given, the
    confidence/agreement check below is computed using only periods anchored
    at BOTH ends by a verified row -- an interpolated row is a deliberate
    estimate, not a confirmed tip, so a period touching one is measurably
    noisier (confirmed on real data: periods anchored by an interpolated row
    showed 2-3x the threshold-sweep spread of fully-detected periods at the
    same column) and shouldn't be trusted to judge whether a column is
    confident. It's still included in the reported median, though -- a
    noisier-but-real sample still improves the point estimate even if it
    shouldn't be the one deciding whether to trust it."""
    if len(aligned_local_indices) < 2:
        return None, None, "too_few_troughs"

    verified_set = set(verified_local_indices) if verified_local_indices is not None else None

    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
    min_dip_depth = max(1.0, MIN_DIP_DEPTH_FRACTION * iqr)
    results = []
    verified_results = []

    for frac in threshold_fractions:
        per_period_duty = []
        verified_duty = []
        for i in range(len(aligned_local_indices) - 1):
            tip, next_tip = aligned_local_indices[i], aligned_local_indices[i + 1]
            period = next_tip - tip
            if period <= 0:
                continue
            local_signal = signal[tip:next_tip + 1]
            b_width = _valley_width(local_signal, frac, min_dip_depth)
            a_width = period - b_width
            a_width = min(period - MIN_RESOLVABLE_DOMAIN_PX, max(MIN_RESOLVABLE_DOMAIN_PX, a_width))
            duty = a_width / period
            per_period_duty.append(duty)
            if verified_set is None or (tip in verified_set and next_tip in verified_set):
                verified_duty.append(duty)

        if len(per_period_duty) < 1:
            continue
        results.append(float(np.median(per_period_duty)) * 100)
        if verified_duty:
            verified_results.append(float(np.median(verified_duty)) * 100)

    if len(results) < 2:
        return None, None, "too_few_troughs"
    # judge confidence from verified-only periods when there are enough of them to be
    # meaningful; otherwise every period in this column touches an interpolated row, so
    # there's nothing more reliable to fall back to than the full set
    confidence_results = verified_results if len(verified_results) >= 2 else results
    if max(confidence_results) - min(confidence_results) > agreement_tolerance:
        return None, None, "low_confidence"
    contrast = float(iqr)
    return float(np.median(results)) / 100, contrast, None


# duty_cycle_for_signal now corrects individual columns for these same
# anomalies (splitting and re-aligning A/B parity around them, see above),
# but that only fixes the SUMMARY statistic for that one column -- it
# doesn't say WHERE the underlying broken/disconnected domain is, or let a
# defect be seen across multiple columns at once. The two functions below
# find and locate individual anomalous pairs directly, entirely separately
# from duty_cycle_for_signal's own per-column correction, for that purpose:
# visualizing/counting broken connections, not just averaging past them.
DOMAIN_ANOMALY_LOW_RATIO = 0.5  # a domain pair shorter than this fraction of its column's local median...
DOMAIN_ANOMALY_HIGH_RATIO = 1.6  # ...or longer than this fraction... flags a likely broken connection right there
DOMAIN_ANOMALY_DILATION = 1  # px -- merges adjacent-column hits from the same physical defect into one cluster.
# Kept tight deliberately: a wider value (tried 3-4) chains distant, vertically-unrelated defects
# together through transitive connectivity in dense regions (confirmed on a real image -- a single
# cluster spanning the whole ROI width and ~40 rows at dilation=4, versus a tight y_span of ~0-2px
# per cluster at dilation=1, matching the physical expectation that one defect sits at one row).
DOMAIN_ANOMALY_MIN_CLUSTER = 3  # minimum points to count as a real defect rather than isolated measurement noise


def find_domain_anomalies(deskewed, x_left, x_right, bottom_row, min_peak_distance, wlen,
                           ratio_low=DOMAIN_ANOMALY_LOW_RATIO, ratio_high=DOMAIN_ANOMALY_HIGH_RATIO,
                           prominence=PROMINENCE, height=ROI_HEIGHT, x_per_row=0.0):
    """Flag individual domain pairs whose length deviates sharply from their
    own column's local median -- a different failure signature from a hard
    column failure (see the note above). Runs its own trough detection
    per column rather than reusing duty_cycle_for_signal's, so a column that
    duty_cycle_for_signal accepts outright can still surface anomalies here.

    Returns a flat list of (x, y, ratio) points in deskewed-image pixel
    space, y being the midpoint between the two troughs bounding the
    anomalous gap. Not clustered/deduplicated -- see cluster_domain_anomalies."""
    points = []
    for x0 in range(x_left, x_right + 1):
        signal = strip_signal(deskewed, x0, bottom_row, height=height, x_per_row=x_per_row)
        troughs, _ = find_peaks(-signal, distance=min_peak_distance, prominence=prominence, wlen=wlen)
        if len(troughs) < 3:
            continue
        refined = np.array([refine_trough_position(signal, i) for i in troughs])
        domain_lengths = np.diff(refined)
        median_len = np.median(domain_lengths)
        if median_len <= 0:
            continue
        ratios = domain_lengths / median_len
        for i, ratio in enumerate(ratios):
            if ratio < ratio_low or ratio > ratio_high:
                mid_signal_idx = (refined[i] + refined[i + 1]) / 2
                y = bottom_row - mid_signal_idx  # strip_signal reverses: index 0 = bottom_row
                x_at_mid = x0 + mid_signal_idx * x_per_row  # follow the same slant strip_signal used
                points.append((float(x_at_mid), float(y), float(ratio)))
    return points


def cluster_domain_anomalies(points, dilation=DOMAIN_ANOMALY_DILATION, min_cluster_size=DOMAIN_ANOMALY_MIN_CLUSTER):
    """Group raw per-column anomaly points into distinct physical defects --
    a single disconnected domain typically shows up across several adjacent
    columns (the same physical gap, sampled repeatedly as the sweep crosses
    it), not as one isolated point. Points are rasterized onto a grid, merged
    via binary dilation, then grouped with connected-component labeling --
    robust to odd-shaped clusters (e.g. a defect whose apparent row drifts a
    little column to column) that a simple pairwise distance threshold would
    fragment.

    Returns a list of dicts (x_center, y_center, x_range, y_range, n_points,
    mean_ratio), sorted by n_points descending. Clusters smaller than
    min_cluster_size are dropped as likely measurement noise rather than a
    real defect."""
    if not points:
        return []

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ratios = np.array([p[2] for p in points])

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    grid = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    grid[(ys - y_min).astype(int), (xs - x_min).astype(int)] = True

    dilated = ndimage.binary_dilation(grid, iterations=dilation)
    labeled, n_labels = ndimage.label(dilated)
    point_labels = labeled[(ys - y_min).astype(int), (xs - x_min).astype(int)]

    clusters = []
    for label in range(1, n_labels + 1):
        mask = point_labels == label
        if mask.sum() < min_cluster_size:
            continue
        cx, cy = xs[mask], ys[mask]
        clusters.append({
            "x_center": float(cx.mean()), "y_center": float(cy.mean()),
            "x_range": (int(cx.min()), int(cx.max())), "y_range": (float(cy.min()), float(cy.max())),
            "n_points": int(mask.sum()), "mean_ratio": float(ratios[mask].mean()),
        })
    clusters.sort(key=lambda c: -c["n_points"])
    return clusters


def segment_columns(x0s, is_failed, smooth_window):
    """Smooth per-column ok/failed status with a rolling-window majority vote
    (so a couple of noisy columns in the middle of an otherwise-clean region
    don't fragment it), then group into contiguous segments. A sample can be
    genuinely non-uniform -- part overpoled, part normal -- and a single
    narrow "representative" slice can miss that split or blend it into one
    misleading average; segmenting the full width instead reports each region
    on its own. Returns a list of (start_x, end_x, is_failed)."""
    is_failed = np.asarray(is_failed, dtype=float)
    kernel = np.ones(smooth_window) / smooth_window
    smoothed_failed = np.convolve(is_failed, kernel, mode="same") > 0.5

    segments = []
    seg_start = 0
    for i in range(1, len(smoothed_failed) + 1):
        if i == len(smoothed_failed) or smoothed_failed[i] != smoothed_failed[seg_start]:
            segments.append((x0s[seg_start], x0s[i - 1], bool(smoothed_failed[seg_start])))
            seg_start = i
    return segments


def rolling_instability(all_dc, window=UNSTABLE_STD_WINDOW, min_valid_fraction=0.5):
    """Per-column rolling-window std of duty cycle values (None entries -- failed
    columns -- are skipped, not treated as 0). A window with too few valid
    entries to trust returns None for that position rather than a misleadingly
    low/high std from a tiny sample."""
    n = len(all_dc)
    half = window // 2
    result = [None] * n
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = [v for v in all_dc[lo:hi] if v is not None]
        if len(window_vals) >= max(3, int((hi - lo) * min_valid_fraction)):
            result[i] = float(np.std(window_vals))
    return result


def summarize_measurements(x0s, duty_cycles, contrasts, contrast_fraction=CONTRAST_FRACTION):
    """Contrast-based column dropping + Tukey IQR outlier rejection + mean/std,
    factored out so it can run once over the whole ROI (backward-compatible
    single-number result) and again per-segment (spatial profile). Returns
    (mean, std, kept_x0s, kept_dcs, is_outlier_mask_over_kept) or all-None if
    nothing survives the contrast filter."""
    x0s = np.asarray(x0s)
    duty_cycles = np.asarray(duty_cycles)
    contrasts = np.asarray(contrasts)

    contrast_cutoff = contrast_fraction * contrasts.max()
    keep = contrasts >= contrast_cutoff
    if not keep.any():
        return None, None, x0s[keep], duty_cycles[keep], np.array([], dtype=bool)

    x0s, duty_cycles = x0s[keep], duty_cycles[keep]

    q1, q3 = np.percentile(duty_cycles, [25, 75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (duty_cycles < lower_fence) | (duty_cycles > upper_fence)

    inliers = duty_cycles[~is_outlier]
    if len(inliers) == 0:
        return None, None, x0s, duty_cycles, is_outlier
    return float(inliers.mean()), float(inliers.std()), x0s, duty_cycles, is_outlier


def compute_duty_cycle(image, x_left, x_right, sweep_x_left=None, sweep_x_right=None, show=True):
    """Deskew image so the grating lines are horizontal, sweep a column range
    (x_left..x_right by default, or sweep_x_left..sweep_x_right if given)
    column by column, drop shadow-degraded columns, and return (mean, std) of
    the surviving per-column duty cycles, as percentages.

    x_left/x_right calibrate the deskew angle, the bottom-line anchor, and the
    domain-length estimate -- these assume relative uniformity across the
    range they're given, so they should stay on a narrow, reliable slice (e.g.
    detect_roi's FINGER_L-based x_left/x_right) even when sweep_x_left/
    sweep_x_right cover a much wider span for the actual measurement. Passing
    a full-width span for calibration too can make the anchor/angle search
    fail outright if that width happens to cross a real transition (e.g. part
    overpoled, part normal) instead of being safely uniform.

    image may be a path to an image file or an already-loaded numpy array."""

    folder = os.path.dirname(os.path.abspath(__file__))
    img = to_gray(image)

    # detect_roi's edge fit can extrapolate past the image bounds for a
    # low-quality edge detection (e.g. mismatched left/right slopes) -- clamp
    # rather than let a negative/out-of-range index silently wrap or crash
    # deep inside cv2.warpAffine
    width = img.shape[1]
    x_left = max(0, min(x_left, width - 2))
    x_right = max(x_left + 1, min(x_right, width - 1))
    sweep_x_left = x_left if sweep_x_left is None else max(0, min(sweep_x_left, width - 2))
    sweep_x_right = x_right if sweep_x_right is None else max(sweep_x_left + 1, min(sweep_x_right, width - 1))

    manual_angle = get_manual_angle(image)
    if manual_angle is not None:
        angle_deg = manual_angle
        print(f"Using saved manual deskew angle: {angle_deg:.2f} deg")
    else:
        angle_deg = find_deskew_angle(img, x_left, x_right)
    deskewed = deskew(img, angle_deg)
    # tip-alignment anchor (mid-domain-B, unambiguous by physical definition)
    # replaces the old flatness-based find_domain_bottom_line, which had no
    # way to tell domain A from domain B -- confirmed to silently invert
    # over/underpoled direction for some images as a result. Deliberately NOT
    # falling back to the flatness search on failure: that method anchors
    # mid-domain-A, the opposite parity duty_cycle_for_signal now assumes, so
    # silently falling back would reintroduce exactly the bug this replaces
    # for whichever images happen to fail tip detection. Failing loudly here
    # (visible as an ERROR in a batch run) is safer than a wrong-but-quiet answer.
    top_row, bottom_search_row = measurement_window(deskewed.shape[0])
    bottom_row = find_tip_alignment_row(deskewed, bottom_search_row)
    print(f"Deskew angle: {angle_deg:.2f} deg, tip-alignment anchor at row {bottom_row}")

    # every tip-alignment row within the measurement window, shared across
    # every column's sweep below -- the metal boundary is a property of the
    # whole image, not of any one interior column, so this only needs
    # computing once. Converted to local signal-index space (index 0 =
    # bottom_row, increasing toward top_row, matching strip_signal's
    # convention) so it can be used directly against any column's signal.
    aligned_rows_global = find_all_tip_alignment_rows(deskewed, top_row, bottom_row)
    detected_rows = set(aligned_rows_global)
    n_detected = len(aligned_rows_global)
    aligned_rows_global = interpolate_missing_tips(aligned_rows_global)
    n_interpolated = len(aligned_rows_global) - n_detected
    aligned_local_indices = sorted(bottom_row - r for r in aligned_rows_global)
    verified_local_indices = [bottom_row - r for r in detected_rows]
    print(f"Found {n_detected} detected tip-alignment rows in the measurement window "
          f"(+{n_interpolated} interpolated for outlier gaps, {len(aligned_local_indices)} total)")

    # ROI strips should follow the metal boundary's own tilt, not assume a
    # perfectly vertical column is correct just because the image was
    # deskewed -- deskewing straightens the domain-grating troughs
    # (horizontal), which isn't guaranteed to leave the boundary perfectly
    # vertical too unless the two tilts are identical. Re-measured directly
    # on the deskewed image (not the pre-deskew angle composed by hand) to
    # avoid a rotation-composition sign mistake -- see detect_boundary_tilt.
    try:
        dir_x, dir_y = detect_boundary_tilt(deskewed)
        x_per_row = -dir_x / dir_y
    except RuntimeError as e:
        print(f"Warning: could not measure boundary tilt for slanted strips ({e}); using straight vertical strips")
        x_per_row = 0.0
    print(f"ROI strip slant: {x_per_row:+.4f} px/row")

    # derive WLEN/MIN_PEAK_DISTANCE from this image's own domain length instead
    # of assuming the pixel scale another image batch happened to be shot at.
    # Sampled from a few columns (not just one) and take the median, so one
    # noisy column can't throw the whole image's parameters off.
    sample_x0s = np.linspace(x_left, x_right, num=min(5, x_right - x_left + 1), dtype=int)
    domain_length = int(np.median([
        estimate_domain_length(strip_signal(deskewed, x0, bottom_row, height=MEASUREMENT_HEIGHT, x_per_row=x_per_row))
        for x0 in sample_x0s
    ]))
    wlen = max(5, int(round(domain_length * WLEN_TO_DOMAIN_RATIO)))
    min_peak_distance = max(1, int(round(domain_length * MIN_DISTANCE_TO_DOMAIN_RATIO)))
    print(f"Estimated domain length: {domain_length}px -> wlen={wlen}, min_peak_distance={min_peak_distance}")

    # sweep every column across the (possibly much wider) measurement range,
    # keeping a result (or failure reason) for every x0 in order -- needed to
    # segment the ROI into contiguous regions rather than only ever seeing the
    # columns that succeeded
    x_left, x_right = sweep_x_left, sweep_x_right
    all_x0 = list(range(x_left, x_right + 1))
    all_dc = []
    all_contrast = []
    all_failed = []
    all_reasons = []  # None for successful columns, else the failure reason -- lets a segment's own
    # local breakdown (not just the whole-image total) decide severely_overpoled vs low_confidence
    failure_counts = {"too_few_troughs": 0, "merged_domain": 0, "low_confidence": 0}

    for x0 in all_x0:
        signal = strip_signal(deskewed, x0, bottom_row, height=MEASUREMENT_HEIGHT, x_per_row=x_per_row)
        dc, contrast, reason = duty_cycle_for_signal(signal, aligned_local_indices, verified_local_indices)
        if dc is not None:
            all_dc.append(dc * 100)
            all_contrast.append(contrast)
            all_failed.append(False)
            all_reasons.append(None)
        else:
            all_dc.append(None)
            all_contrast.append(None)
            all_failed.append(True)
            all_reasons.append(reason)
            failure_counts[reason] += 1

    total = len(all_x0)
    n_valid_total = total - sum(all_failed)
    print(f"Computed duty cycle for {n_valid_total}/{total} x-positions")
    print(f"Failed columns: {failure_counts['too_few_troughs']} too few troughs, "
          f"{failure_counts['merged_domain']} merged-domain (likely overpoled), "
          f"{failure_counts['low_confidence']} low-confidence (unstable across prominence thresholds)")

    # isolated broken/disconnected domain connections -- invisible to the
    # column-level checks above by design (see the note on
    # DOMAIN_ANOMALY_LOW_RATIO), so found independently here and reported as
    # their own diagnostic rather than folded into the ok/problem segmentation.
    anomaly_points = find_domain_anomalies(deskewed, x_left, x_right, bottom_row, min_peak_distance, wlen,
                                            height=MEASUREMENT_HEIGHT, x_per_row=x_per_row)
    domain_anomalies = cluster_domain_anomalies(anomaly_points)
    print(f"Domain anomaly scan: {len(anomaly_points)} raw flagged pairs -> "
          f"{len(domain_anomalies)} distinct defect location(s)")
    for c in domain_anomalies:
        print(f"  defect near x={c['x_center']:.0f}, y={c['y_center']:.0f} "
              f"({c['n_points']} pts, mean ratio {c['mean_ratio']:.2f})")

    if n_valid_total == 0:
        # every column failed -- report a clear classification instead of
        # raising, so a batch run records this as a real (if imprecise)
        # outcome rather than stopping on an exception.
        #
        # too_few_troughs/merged_domain both mean the same thing in practice:
        # no boundary trough survives anywhere in the ROI, the more extreme
        # end of a collapse (validated against real samples: images confirmed
        # severely overpoled failed via too_few_troughs, not merged_domain).
        # low_confidence is different -- troughs WERE found, they just didn't
        # agree across prominence thresholds (see CONFIDENCE_AGREEMENT_TOLERANCE
        # above), which reads as "contrast too poor to trust", not "collapsed".
        # A column majority-low_confidence gets its own honest label instead
        # of being folded into the (calibrated-for-a-different-failure)
        # severely_overpoled bucket.
        if failure_counts["low_confidence"] >= failure_counts["too_few_troughs"] + failure_counts["merged_domain"]:
            classification = "low_confidence"
            message = (
                f"Unmeasurable: {failure_counts['low_confidence']}/{total} columns found troughs but "
                f"disagreed across prominence thresholds (contrast too poor to trust a specific ratio), "
                f"and none of the rest cleared even a lenient trough-detection pass "
                f"({failure_counts['merged_domain']}/{total} merged-domain, "
                f"{failure_counts['too_few_troughs']}/{total} too few troughs). Unlike severely_overpoled, "
                f"this isn't evidence of a direction -- it's evidence the image quality in this region "
                f"can't support a precise measurement either way."
            )
        else:
            classification = "severely_overpoled"
            message = (
                f"Severely overpoled: no boundary trough survived anywhere in the ROI "
                f"({failure_counts['merged_domain']}/{total} columns showed a merged-domain signature, "
                f"{failure_counts['too_few_troughs']}/{total} had too few troughs to even compare, "
                f"{failure_counts['low_confidence']}/{total} were low-confidence) -- "
                f"a precise ratio can't be computed, so this direction is inferred, not measured: prior "
                f"hand-confirmed overpoled samples failed this exact way (see MERGE_RATIO_THRESHOLD/"
                f"too_few_troughs above). An equally severe underpoled sample hasn't yet been confirmed to "
                f"fail the same way, so treat 'overpoled' here as a working assumption until one is checked. "
                f"(If this seems wrong for this image, also check focus/contrast/ROI placement -- a bad "
                f"image can fail the same way.)"
            )
        print(f"RESULT: {classification} -- {message}")
        diagnostics = {
            "n_total": total,
            "n_valid": 0,
            "n_merged_domain": failure_counts["merged_domain"],
            "n_too_few_troughs": failure_counts["too_few_troughs"],
            "n_low_confidence": failure_counts["low_confidence"],
            "merged_fraction": failure_counts["merged_domain"] / total,
            "problem_fraction": 1.0,
            "classification": classification,
            "message": message,
            "segments": [{
                "x_start": x_left, "x_end": x_right, "classification": classification,
                "mean": None, "std": None, "n_valid": 0, "n_total": total,
            }],
            "domain_anomalies": domain_anomalies,
            "n_domain_anomalies": len(domain_anomalies),
        }
        return None, None, diagnostics

    # a column can "succeed" (find_peaks still finds troughs, still computes a
    # ratio) while still being wrong -- a mildly overpoled/underpoled region
    # can swing wildly column to column instead of failing outright, averaging
    # to a mean deceptively close to 50%. Flag that instability too, not just
    # hard failures, before segmenting.
    local_std = rolling_instability(all_dc)
    all_unstable = [not f and std is not None and std > UNSTABLE_STD_THRESHOLD
                     for f, std in zip(all_failed, local_std)]
    all_problem = [f or u for f, u in zip(all_failed, all_unstable)]

    # segment the swept width into contiguous ok/problem regions -- a sample
    # can be genuinely non-uniform (e.g. part overpoled, part normal), and
    # blending both into one average hides exactly the thing that matters.
    # Smoothing window scales with domain_length so a handful of isolated
    # noisy columns don't fragment an otherwise-consistent region. Kept
    # smaller than earlier tuning (domain_length*3): all_problem here is
    # already built from rolling_instability's own window (UNSTABLE_STD_WINDOW
    # = 15), so a second wide smoothing pass on top compounds into real lag --
    # confirmed on a real image where the segment boundary landed ~13px past
    # where the raw per-column data had already gone clean.
    smooth_window = max(9, int(domain_length * 1.5))
    raw_segments = segment_columns(all_x0, all_problem, smooth_window)
    print(f"Segmented into {len(raw_segments)} region(s): "
          + ", ".join(f"x={s}-{e} ({'problem' if f else 'ok'})" for s, e, f in raw_segments))

    segments = []
    for seg_start, seg_end, seg_is_problem in raw_segments:
        i0, i1 = seg_start - x_left, seg_end - x_left + 1
        seg_failed_flags = all_failed[i0:i1]
        seg_reasons = all_reasons[i0:i1]
        seg_x0s = [x for x, f in zip(all_x0[i0:i1], seg_failed_flags) if not f]
        seg_dcs = [d for d, f in zip(all_dc[i0:i1], seg_failed_flags) if not f]
        seg_contrasts = [c for c, f in zip(all_contrast[i0:i1], seg_failed_flags) if not f]
        seg_total = seg_end - seg_start + 1
        n_hard_failed = sum(seg_failed_flags)

        if len(seg_dcs) == 0:
            # see the matching whole-image logic above: low_confidence means
            # troughs were found but didn't agree, which isn't evidence of
            # overpoling the way a genuine collapse (too_few_troughs/
            # merged_domain) is -- keep the two labeled separately here too.
            n_low_conf = seg_reasons.count("low_confidence")
            n_collapsed = seg_reasons.count("too_few_troughs") + seg_reasons.count("merged_domain")
            seg_classification = "low_confidence" if n_low_conf >= n_collapsed else "severely_overpoled"
            segments.append({"x_start": seg_start, "x_end": seg_end, "classification": seg_classification,
                              "mean": None, "std": None, "n_valid": 0, "n_total": seg_total})
            continue

        mean, std, *_ = summarize_measurements(seg_x0s, seg_dcs, seg_contrasts)
        if not seg_is_problem:
            classification = "ok" if mean is not None else "unmeasurable"
        elif n_hard_failed / seg_total > 0.3:
            # direction read from whatever ratios DID survive in this mostly-
            # failed segment, not from the failure mode itself -- see
            # classify_direction's docstring for why the sign is trustworthy
            # even here
            direction = classify_direction(seg_dcs)
            classification = {
                "overpoled": "partially_overpoled",
                "underpoled": "partially_underpoled",
                "bidirectional": "partially_bidirectional",
            }.get(direction, "partially_indeterminate")
        else:
            # every column here technically produced a ratio, but they swing
            # wildly (see UNSTABLE_STD_THRESHOLD) -- a real overpoled/underpoled
            # region, just not severe enough to erase troughs outright. Which
            # direction depends on which way the surviving ratios actually lean.
            direction = classify_direction(seg_dcs)
            classification = {
                "overpoled": "unstable_likely_overpoled",
                "underpoled": "unstable_likely_underpoled",
                "bidirectional": "unstable_bidirectional",
            }.get(direction, "unstable_indeterminate")
        segments.append({
            "x_start": seg_start, "x_end": seg_end,
            "classification": classification,
            "mean": mean, "std": std, "n_valid": len(seg_dcs), "n_total": seg_total,
        })

    for seg in segments:
        if seg["mean"] is not None:
            print(f"  Segment x={seg['x_start']}-{seg['x_end']}: {seg['mean']:.2f}% +/- {seg['std']:.2f}% "
                  f"({seg['n_valid']}/{seg['n_total']} valid)")
        else:
            print(f"  Segment x={seg['x_start']}-{seg['x_end']}: {seg['classification']} "
                  f"({seg['n_valid']}/{seg['n_total']} valid)")

    # overall (mean, std) is built from "ok" segments plus any "problem"
    # segment where enough of its own columns still measured confidently --
    # a segment being flagged unstable/bidirectional doesn't mean every
    # column in it failed; discarding the whole segment threw away
    # individually-trustworthy measurements (confirmed on a real image: a
    # segment with 6/23 valid columns, each of which passed the exact same
    # per-column confidence check as any "ok" column, was being excluded
    # entirely). The segment's own classification label is untouched here --
    # still flags real regional instability -- this only changes whether its
    # valid columns count toward the single overall number. Hard failures
    # and genuinely sparse segments (below the fraction) are still excluded,
    # so a mostly-collapsed region still can't sneak into the average.
    include_ranges = [
        (s["x_start"], s["x_end"]) for s in segments
        if s["classification"] == "ok"
        or (s["n_total"] > 0 and s["n_valid"] / s["n_total"] >= SEGMENT_INCLUDE_MIN_VALID_FRACTION)
    ]
    in_include_segment = lambda x0: any(lo <= x0 <= hi for lo, hi in include_ranges)  # noqa: E731

    ok_x0s = [x for x, f in zip(all_x0, all_failed) if not f and in_include_segment(x)]
    ok_dcs = [d for x, d, f in zip(all_x0, all_dc, all_failed) if not f and in_include_segment(x)]
    ok_contrasts = [c for x, c, f in zip(all_x0, all_contrast, all_failed) if not f and in_include_segment(x)]

    if len(ok_dcs) == 0:
        mean_duty_cycle, std_duty_cycle = None, None
        kept_x0s, kept_dcs, is_outlier = np.array([]), np.array([]), np.array([], dtype=bool)
    else:
        mean_duty_cycle, std_duty_cycle, kept_x0s, kept_dcs, is_outlier = summarize_measurements(
            ok_x0s, ok_dcs, ok_contrasts
        )
        kept_x0s = np.asarray(kept_x0s)
        kept_dcs = np.asarray(kept_dcs)

    roi_vis = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    top_shift = (bottom_row - top_row) * x_per_row  # follow the same slant the strips actually used
    for seg in segments:
        color = SEGMENT_COLOR_BGR[classification_direction_key(seg["classification"])]
        quad = np.array([
            [seg["x_start"], bottom_row],
            [seg["x_end"], bottom_row],
            [int(round(seg["x_end"] + top_shift)), top_row],
            [int(round(seg["x_start"] + top_shift)), top_row],
        ], dtype=np.int32)
        cv2.polylines(roi_vis, [quad], isClosed=True, color=color, thickness=1)
    # green circles mark individual broken/disconnected domain locations --
    # a distinct signal from the segment rectangles above, which only show
    # column-range-level classification and can miss isolated defects entirely
    for c in domain_anomalies:
        cv2.circle(roi_vis, (int(round(c["x_center"])), int(round(c["y_center"]))),
                    max(3, min(10, c["n_points"])), (0, 255, 0), 1)
    roi_vis_path = os.path.join(folder, "..", "outputs", "duty_profile_roi.png")
    cv2.imwrite(roi_vis_path, roi_vis)
    print(f"Saved ROI visualization to {roi_vis_path}")

    fig, ax = plt.subplots(figsize=(12, 5))
    present_keys = set()
    for seg in segments:
        key = classification_direction_key(seg["classification"])
        if key != "ok":
            ax.axvspan(seg["x_start"], seg["x_end"], color=SEGMENT_COLOR_MPL.get(key, "red"), alpha=0.12)
            present_keys.add(key)

    # mark every segment boundary with its actual x-coordinate -- lets the ROI
    # image (which shows the same boundaries as box edges) and this plot be
    # read side by side, instead of having to guess which shaded block
    # corresponds to which box
    boundary_xs = sorted({b for seg in segments for b in (seg["x_start"], seg["x_end"])})
    for bx in boundary_xs:
        ax.axvline(bx, color="0.4", linestyle=":", linewidth=0.8, zorder=0)
        ax.annotate(str(bx), xy=(bx, 1), xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", rotation=90, va="top", ha="left", fontsize=7, color="0.3")

    # raw per-column values for EVERY successfully-measured column, including
    # ones inside problem segments -- otherwise a shaded region shows up as an
    # empty gap with no way to see what the underlying signal actually looks
    # like (erratic swings vs. a clean one-sided bias vs. a hard gap)
    raw_x0s = [x for x, d in zip(all_x0, all_dc) if d is not None]
    raw_dcs = [d for d in all_dc if d is not None]
    raw_line, = ax.plot(raw_x0s, raw_dcs, color="0.75", linewidth=1, marker=".", markersize=3, zorder=1,
                          label="all measured columns")
    line, = ax.plot(kept_x0s, kept_dcs, marker="o", markersize=3, zorder=2, label="ok (kept)")

    legend_handles = [raw_line, line] + [
        Patch(facecolor=SEGMENT_COLOR_MPL.get(k, "red"), alpha=0.3, label=k.replace("_", " "))
        for k in sorted(present_keys)
    ]
    if len(kept_x0s) and is_outlier.any():
        outlier_line, = ax.plot(kept_x0s[is_outlier], kept_dcs[is_outlier], "rx", markersize=8, label="excluded outlier")
        legend_handles.append(outlier_line)
    if domain_anomalies:
        for c in domain_anomalies:
            defect_line = ax.axvline(c["x_center"], color="green", linestyle="--", linewidth=1, alpha=0.6)
        defect_line.set_label(f"{len(domain_anomalies)} broken-domain location(s)")
        legend_handles.append(defect_line)
    ax.legend(handles=legend_handles)

    ax.set_xlabel("Strip starting x position (px)")
    ax.set_ylabel("Duty cycle (%)")
    ax.set_title("Duty cycle across full ROI width (shaded = non-uniform region, color = direction)")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(folder, "..", "outputs", "duty_cycle_profile.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    if show:
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        def update_annot(index):
            x, y = line.get_data()
            px, py = x[index], y[index]
            annot.xy = (px, py)
            annot.set_text(f"x={px}\nduty={py:.2f}%")

        def on_hover(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            contains, details = line.contains(event)
            if contains:
                update_annot(details["ind"][0])
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.show()
    else:
        plt.close(fig)

    merged_fraction = failure_counts["merged_domain"] / total
    # fraction of the swept width NOT covered by an "ok" segment -- the single
    # explicit non-uniformity signal a caller (e.g. the BO loop) can penalize
    # directly, instead of it only being visible by digging into 'segments'
    ok_width = sum(s["x_end"] - s["x_start"] + 1 for s in segments if s["classification"] == "ok")
    problem_fraction = 1.0 - (ok_width / total)
    has_ok_segment = any(s["classification"] == "ok" for s in segments)
    has_problem_segment = any(s["classification"] != "ok" for s in segments)

    if has_ok_segment and has_problem_segment:
        classification = "spatially_varying"
    elif has_problem_segment:
        # every successfully-measured column landed in a problem segment --
        # not a total hard-failure (that's caught earlier as severely_overpoled),
        # but nothing here was stable/clean enough to trust as "the" measurement.
        # Direction is a width-weighted rollup of the per-segment directions
        # above, not a fresh guess -- if segments disagree on direction, that's
        # reported as "mixed_directions" rather than silently picking one.
        problem_widths = {}
        for s in segments:
            key = classification_direction_key(s["classification"])
            if key == "ok":
                continue
            w = s["x_end"] - s["x_start"] + 1
            problem_widths[key] = problem_widths.get(key, 0) + w
        classification = next(iter(problem_widths)) if len(problem_widths) == 1 else "mixed_directions"
    else:
        classification = "ok"

    direction_messages = {
        "overpoled": "No region of this sample measured as clean/stable -- every successfully-measured "
                      "column fell in an overpoled segment (domain A oversized relative to B). mean/std "
                      "are unavailable; see 'segments'.",
        "underpoled": "No region of this sample measured as clean/stable -- every successfully-measured "
                      "column fell in an underpoled segment (domain A undersized relative to B). mean/std "
                      "are unavailable; see 'segments'.",
        "bidirectional": "No region of this sample measured as clean/stable -- surviving columns swing "
                          "between overpoled and underpoled rather than leaning one way. mean/std are "
                          "unavailable; see 'segments'.",
        "mixed_directions": "No region of this sample measured as clean/stable, and different regions lean "
                             "different directions (over/under/unstable) -- see 'segments' for the "
                             "per-region breakdown.",
        "severely_overpoled": "Severely overpoled: no boundary trough survived anywhere in the ROI, so no "
                               "ratio could be read to confirm direction -- classified overpoled based on "
                               "prior hand-confirmed samples that failed this exact way (see "
                               "MERGE_RATIO_THRESHOLD/too_few_troughs above). An equally severe underpoled "
                               "sample has not yet been confirmed to fail the same way; treat this label "
                               "with that caveat until one is checked.",
        "indeterminate": "No region of this sample measured as clean/stable, and the surviving ratios don't "
                          "lean clearly enough in either direction to call it over- or under-poled. mean/std "
                          "are unavailable; see 'segments'.",
    }

    diagnostics = {
        "n_total": total,
        "n_valid": n_valid_total,
        "n_merged_domain": failure_counts["merged_domain"],
        "n_too_few_troughs": failure_counts["too_few_troughs"],
        "n_low_confidence": failure_counts["low_confidence"],
        "merged_fraction": merged_fraction,
        "problem_fraction": problem_fraction,
        "classification": classification,
        "segments": segments,
        "domain_anomalies": domain_anomalies,
        "n_domain_anomalies": len(domain_anomalies),
        "message": (
            f"Sample varies spatially: {len(segments)} region(s) detected, not uniform across the ROI -- "
            f"reported mean/std draw from the 'ok' region(s) plus any flagged region where at least "
            f"{SEGMENT_INCLUDE_MIN_VALID_FRACTION:.0%} of its own columns still measured confidently "
            f"({100 * (1 - problem_fraction):.0f}% of the width is 'ok' outright); see 'segments' for the "
            f"full per-region breakdown, including which flagged regions contributed."
            if classification == "spatially_varying" else direction_messages.get(classification, "")
        ),
    }
    return mean_duty_cycle, std_duty_cycle, diagnostics


if __name__ == "__main__":
    # standalone testing values -- these are edge.py's detected ROI bounds for
    # DEFAULT_IMAGE, hand-copied; run via main.py to derive them automatically instead
    X_LEFT = 307
    X_RIGHT = 468

    duty_cycle, sd, diagnostics = compute_duty_cycle(DEFAULT_IMAGE, X_LEFT, X_RIGHT, show=True)
    print(f"({duty_cycle:.2f}, {sd:.2f}) {diagnostics}")
