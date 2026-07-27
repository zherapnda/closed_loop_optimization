import cv2
import os

# ---- change this to your image filename ----
IMAGE_NAME = "image1.png"
# --------------------------------------------

# find the image in the same folder as this script
folder = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(folder, IMAGE_NAME)

img = cv2.imread(path)
if img is None:
    print(f"ERROR: could not find {path}")
    print("Make sure the image is in the same folder as this script.")
    exit()

# make the image bigger so it's easier to click accurately
scale = 2
img_big = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))

clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append(y)  # we care about y (vertical position), not x
        # draw a horizontal red line where you clicked
        cv2.line(img_big, (0, y), (img_big.shape[1], y), (0, 0, 255), 1)
        cv2.imshow("measure stripe - click top then bottom of ONE bright stripe", img_big)

        if len(clicks) == 1:
            print(f"First click at y={y//scale}px  -- now click the BOTTOM edge of the same stripe")

        if len(clicks) == 2:
            stripe_height_pixels = abs(clicks[1] - clicks[0]) // scale
            print(f"Second click at y={y//scale}px")
            print(f"\nStripe height = {stripe_height_pixels} pixels")
            print(f"Suggested HALF_WIDTH = {stripe_height_pixels // 2}")
            print("\nPress Q to quit.")

cv2.imshow("measure stripe - click top then bottom of ONE bright stripe", img_big)
cv2.setMouseCallback("measure stripe - click top then bottom of ONE bright stripe", on_click)

print("=== Stripe Measurer ===")
print(f"Image: {IMAGE_NAME}")
print("Instructions:")
print("  1. Look for a light gray horizontal stripe in the middle of the image")
print("  2. Click its TOP edge")
print("  3. Click its BOTTOM edge")
print("  -> The script will print the height in pixels")
print()
print("Press Q to quit at any time.")

cv2.waitKey(0)
cv2.destroyAllWindows()