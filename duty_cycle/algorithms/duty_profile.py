import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

from loader import to_gray


DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image2.png")
ROI_HEIGHT = 400

MIN_PEAK_DISTANCE = 2
PROMINENCE = 2
WLEN = 21  # samples -- bounds prominence/width search so it can't reach past neighboring peaks

# a column is dropped from the ROI if its ridge contrast (median peak prominence)
# falls below this fraction of the best contrast seen anywhere in the sweep --
# catches shadow-degraded columns even though the degradation is a smooth
# gradient with no single boundary to detect
CONTRAST_FRACTION = 0.3


def strip_signal(img, rotation_deg, x0, height):
    M = cv2.getRotationMatrix2D((x0, 0), rotation_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    crop = rotated[0:height, x0:x0 + 1]
    return crop.flatten().astype(float)


def duty_cycle_for_signal(signal):
    peaks, _ = find_peaks(signal, distance=MIN_PEAK_DISTANCE, prominence=PROMINENCE, wlen=WLEN)
    if len(peaks) < 2:
        return None, None

    periods = np.diff(peaks)
    widths = peak_widths(signal, peaks, rel_height=0.5, wlen=WLEN)[0]
    per_cycle_duty = widths[:-1] / periods
    contrast = float(np.median(peak_prominences(signal, peaks, wlen=WLEN)[0]))
    return float(np.median(per_cycle_duty)), contrast


def compute_duty_cycle(image, slope, x_left, x_right, roi_height=ROI_HEIGHT, show=True):
    """Sweep the ROI (x_left..x_right at the top row, tilted by slope) column by
    column, drop shadow-degraded columns, and return (mean, std) of the
    surviving per-column duty cycles, as percentages. image may be a path to an
    image file or an already-loaded numpy array."""

    folder = os.path.dirname(os.path.abspath(__file__))
    img = to_gray(image)

    angle_line_deg = np.degrees(np.arctan(slope))
    rotation_deg = angle_line_deg + 90

    x_positions = []
    duty_cycles = []
    contrasts = []

    for x0 in range(x_left, x_right + 1):
        signal = strip_signal(img, rotation_deg, x0, roi_height)
        dc, contrast = duty_cycle_for_signal(signal)
        if dc is not None:
            x_positions.append(x0)
            duty_cycles.append(dc * 100)
            contrasts.append(contrast)

    total = x_right - x_left + 1
    print(f"Computed duty cycle for {len(x_positions)}/{total} x-positions")

    x_positions = np.array(x_positions)
    duty_cycles = np.array(duty_cycles)
    contrasts = np.array(contrasts)

    contrast_cutoff = CONTRAST_FRACTION * contrasts.max()
    keep = contrasts >= contrast_cutoff

    print(f"Contrast cutoff: {contrast_cutoff:.2f} (best seen: {contrasts.max():.2f})")
    print(f"Dropped {(~keep).sum()} shadow-degraded columns, kept x={x_positions[keep].min()}-{x_positions[keep].max()}")

    x_positions = x_positions[keep]
    duty_cycles = duty_cycles[keep]

    roi_left, roi_right = int(x_positions.min()), int(x_positions.max())
    dx_bottom = roi_height / slope

    roi_box = np.array([
        [roi_left, 0],
        [roi_right, 0],
        [int(round(roi_right + dx_bottom)), roi_height],
        [int(round(roi_left + dx_bottom)), roi_height],
    ], dtype=np.int32)

    roi_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(roi_vis, [roi_box], isClosed=True, color=(0, 255, 255), thickness=2)

    roi_vis_path = os.path.join(folder, "..", "outputs", "duty_profile_roi.png")
    cv2.imwrite(roi_vis_path, roi_vis)
    print(f"Saved ROI visualization to {roi_vis_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x_positions, duty_cycles, marker="o", markersize=3)
    ax.set_xlabel("Strip starting x position (px)")
    ax.set_ylabel("Duty cycle (%)")
    ax.set_title("Duty cycle across ROI")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(folder, "..", "outputs", "duty_cycle_profile.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    mean_duty_cycle = float(duty_cycles.mean())
    std_duty_cycle = float(duty_cycles.std())

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

    return mean_duty_cycle, std_duty_cycle


if __name__ == "__main__":
    # standalone testing values -- these are edge.py's detected ROI bounds for
    # DEFAULT_IMAGE, hand-copied; run via main.py to derive them automatically instead
    SLOPE = -29.76675
    X_LEFT = 307
    X_RIGHT = 468

    duty_cycle, sd = compute_duty_cycle(DEFAULT_IMAGE, SLOPE, X_LEFT, X_RIGHT, show=True)
    print(f"({duty_cycle:.2f}, {sd:.2f})")
