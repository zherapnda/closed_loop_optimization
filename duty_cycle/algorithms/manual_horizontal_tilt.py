import os
import cv2
import numpy as np

IMAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image1.png")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "manual_tilt.png")

points = []


def draw_lines(img, pts):
    if len(pts) == 0:
        return

    for (x, y) in pts:
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

    if len(pts) < 2:
        return

    (x1, y1), (x2, y2) = pts
    y_horiz = int(round((y1 + y2) / 2.0))
    x_start = min(x1, x2)
    x_end = max(x1, x2)

    # actual line between the clicked points
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # horizontal comparator line
    cv2.line(img, (x_start, y_horiz), (x_end, y_horiz), (255, 0, 0), 2)
    cv2.putText(
        img,
        f"horizontal y={y_horiz}",
        (x_start, max(0, y_horiz - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    tilt_text = f"tilt={angle_deg:.2f} deg"
    cv2.putText(
        img,
        tilt_text,
        (x_start, min(img.shape[0] - 10, y_horiz + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            points = []
        points.append((x, y))


if __name__ == "__main__":
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {IMAGE_FILE}")

    window_name = "Manual Horizontal Tilt"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click two points to draw a horizontal comparison line.")
    print("Press 'r' to reset, 's' to save, or 'q' to quit.")

    while True:
        display = img.copy()
        draw_lines(display, points)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            points = []
        if key == ord("s") and len(points) == 2:
            saved = display.copy()
            cv2.imwrite(OUTPUT_FILE, saved)
            print(f"Saved manual tilt image to {OUTPUT_FILE}")

    cv2.destroyAllWindows()
