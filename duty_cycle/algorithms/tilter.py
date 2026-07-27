# tilt.py
# Automatically detects the tilt angle of the fingers in an image
# using the Hough Line Transform

import cv2
import numpy as np


def detect_tilt_angle(gray):
    """
    Takes a grayscale image.
    Returns the tilt angle of the fingers in degrees.
    """

    # step 1: blur slightly to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # step 2: edge detection — finds where brightness changes sharply
    # these sharp changes = the dark boundaries between fingers
    edges = cv2.Canny(blurred,
                      threshold1=30,    # lower = detects more edges
                      threshold2=100)   # upper = keeps only strong edges

    # step 3: Hough Line Transform — finds straight lines in the edges
    lines = cv2.HoughLinesP(edges,
                             rho=1,             # precision in pixels
                             theta=np.pi/180,   # precision in degrees (1 degree steps)
                             threshold=50,      # minimum votes for a line to count
                             minLineLength=50,  # ignore lines shorter than this (pixels)
                             maxLineGap=10)     # allow small gaps in lines

    if lines is None:
        print("WARNING: no lines detected — returning angle 0")
        return 0.0

      # step 4: calculate the angle of every detected line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        # only keep angles that are nearly vertical
        # (fingers run nearly vertically, so angle should be close to 90 or -90)
        if abs(angle) > 60:
            angles.append(angle)

    if len(angles) == 0:
        print("WARNING: no vertical lines detected — returning angle 0")
        return 0.0

    # step 5: take the median angle — more robust than average
    # median ignores outlier lines that dont belong to fingers
    tilt_angle = float(np.median(angles))

    return tilt_angle


# ---- test block: only runs when you run this file directly ----
if __name__ == "__main__":
    import os
    from loader import grayer

    IMAGE_NAME = "image1.png"
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, "images", IMAGE_NAME)

    gray = grayer(path)
    angle = detect_tilt_angle(gray)
    print(f"Detected tilt angle:              {angle:.2f} degrees")
    print(f"Expected (manual measurement):    -87.55 degrees")