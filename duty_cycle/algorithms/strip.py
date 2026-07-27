
import os
import cv2
import numpy as np


IMAGE_NAME = "image2.png"
STRIP_WIDTH = 1
STRIP_HEIGHT = 400
ZOOM = 10
SLOPE = -29.76675


folder = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(folder, "images", IMAGE_NAME)

img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Unable to read image at: {path}")



x0 = 375
y0 = 0
x1 = x0 + STRIP_WIDTH
y1 = 0 + STRIP_HEIGHT

dy_line = STRIP_HEIGHT
dx_line = dy_line / SLOPE
end_x = int(round(x0 + dx_line))
end_y = int(round(y0 + dy_line))

traced = img.copy()
cv2.line(traced, (x0, y0), (end_x, end_y), (0, 0, 255), 2)

angle_line_deg = np.degrees(np.arctan(SLOPE))
rotation_deg = angle_line_deg + 90

M = cv2.getRotationMatrix2D((x0, y0), rotation_deg, 1.0)
rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

crop = rotated[y0:y1, x0:x1].copy()


zoomed = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), 
                    interpolation=cv2.INTER_NEAREST)


cv2.imshow("middle strip zoomed in -- press Q to quit", zoomed)
cv2.imshow("original -- red line traces the strip", traced)
cv2.waitKey(0)
cv2.destroyAllWindows()


output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "middle_strip.png")
cv2.imwrite(output_path, crop)
print(f"Saved marked image to {output_path}")

traced_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "strip_t.png")
cv2.imwrite(traced_path, traced)
print(f"Saved traced-line image to {traced_path}")