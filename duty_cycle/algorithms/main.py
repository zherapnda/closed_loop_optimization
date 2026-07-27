import os
from edge import detect_roi
from duty_profile import compute_duty_cycle

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "image2.png")


def main(image, roi_height=400, show=False):
    """image may be a path to an image file or an already-loaded numpy array
    (grayscale or color). Returns (duty_cycle, sd) as percentages."""
    roi = detect_roi(image, roi_height=roi_height, show=show)
    duty_cycle, sd = compute_duty_cycle(
        image,
        roi["slope"],
        roi["x_left"],
        roi["x_right"],
        roi_height=roi_height,
        show=show,
    )
    return duty_cycle/100, sd/100


if __name__ == "__main__":
    result = main(DEFAULT_IMAGE)
    print(result)
