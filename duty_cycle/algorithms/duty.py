import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

from loader import grayer


image = "middle_strip.png"
MIN_PEAK_DISTANCE = 2
PROMINENCE = 2  # grayscale units -- tune based on the printed prominences below
WLEN = 21  # samples -- bounds prominence/width search so it can't reach past neighboring peaks

tracefolder = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(tracefolder, "..", "outputs", image)

strip = cv2.imread(path)
if strip is None:
    raise FileNotFoundError(f"Unable to read image at: {path}")


gr = grayer(path)
signal = gr.mean(axis=1)

peaks, _ = find_peaks(signal, distance=MIN_PEAK_DISTANCE, prominence=PROMINENCE, wlen=WLEN)
troughs, _ = find_peaks(-signal, distance=MIN_PEAK_DISTANCE, prominence=PROMINENCE, wlen=WLEN)

print(f"Peak positions: {peaks}")
print(f"Peak prominences: {peak_prominences(signal, peaks, wlen=WLEN)[0].round(1)}")

if len(peaks) < 2:
    raise ValueError(
        f"Only found {len(peaks)} peak(s) -- need at least 2 to measure a period. "
        "Try lowering PROMINENCE."
    )

periods = np.diff(peaks)
widths, width_heights, left_ips, right_ips = peak_widths(signal, peaks, rel_height=0.5, wlen=WLEN)

# pair each cycle's on-width with its own period, rather than averaging widths
# and periods separately (a ratio of means would let one long/short cycle
# distort the rest)
per_cycle_duty = widths[:-1] / periods

print(f"Detected {len(peaks)} peaks, {len(troughs)} troughs")
print(f"Per-cycle periods (px): {periods}")
print(f"Per-cycle on-widths (px): {widths[:-1].round(2)}")
print(f"Per-cycle duty cycles: {per_cycle_duty.round(3)}")
print(f"Median period: {np.median(periods):.2f} px")
print(f"Median duty cycle: {np.median(per_cycle_duty):.2%}")
print(f"Mean duty cycle: {per_cycle_duty.mean():.2%}")

ZOOM_Y = 6
STRIP_DISPLAY_WIDTH = 150
MARGIN = 90

height = gr.shape[0] * ZOOM_Y
strip_zoomed = cv2.resize(gr, (STRIP_DISPLAY_WIDTH, height), interpolation=cv2.INTER_NEAREST)
strip_bgr = cv2.cvtColor(strip_zoomed, cv2.COLOR_GRAY2BGR)

canvas = np.full((height, STRIP_DISPLAY_WIDTH + 2 * MARGIN, 3), 255, dtype=np.uint8)
canvas[:, MARGIN:MARGIN + STRIP_DISPLAY_WIDTH] = strip_bgr

for r in peaks:
    y = int(r * ZOOM_Y + ZOOM_Y / 2)
    cv2.line(canvas, (0, y), (MARGIN, y), (0, 0, 255), 1)
    cv2.putText(canvas, str(r), (5, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

for r in troughs:
    y = int(r * ZOOM_Y + ZOOM_Y / 2)
    cv2.line(canvas, (MARGIN + STRIP_DISPLAY_WIDTH, y), (canvas.shape[1], y), (255, 0, 0), 1)
    cv2.putText(canvas, str(r), (MARGIN + STRIP_DISPLAY_WIDTH + 5, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

annotated_path = os.path.join(tracefolder, "..", "outputs", "strip_annotated_rows.png")
cv2.imwrite(annotated_path, canvas)
print(f"Saved annotated strip to {annotated_path}")

plt.figure(figsize=(14, 6))
plt.plot(signal, linewidth=1, label="brightness")
plt.plot(peaks, signal[peaks], "rx", markersize=8, label="peaks (on)")
plt.plot(troughs, signal[troughs], "bo", markersize=4, label="troughs (off)")
plt.hlines(width_heights, left_ips, right_ips, color="green", linewidth=2, label="on-width (half-prominence)")
plt.xlabel("Row index (top of strip = 0)")
plt.ylabel("Grayscale value")
plt.title("Peak/trough/on-width check")
plt.legend()
plt.grid(True, alpha=0.3)

viz_path = os.path.join(tracefolder, "..", "outputs", "duty_cycle_visual_check.png")
plt.savefig(viz_path, dpi=150)
print(f"Saved visual check to {viz_path}")

plt.show()
