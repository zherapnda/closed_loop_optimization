import cv2
import numpy as np

def grayer(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at: {path}.\nCheck that the file exists and is a valid image format.")
    return img


def to_gray(image):
    """Accept either a path to an image file or an already-loaded numpy array
    (grayscale or color) and return a 2D grayscale array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayer(image)


def to_color(image):
    """Accept either a path to an image file or an already-loaded numpy array
    (grayscale or color) and return a 3-channel BGR array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Unable to read image at: {image}.\nCheck that the file exists and is a valid image format.")
    return img