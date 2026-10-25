# Exercise 5.18 Image inpainting with different photos and several ROIs
# Usage: python exercise5.18.py                      (church.jpeg, draw the ROIs yourself)
#        python exercise5.18.py --image fruits.jpg   (another photo)
#        python exercise5.18.py --auto               (preset ROIs on church.jpeg, no mouse needed)
# Draw a rectangle, press ENTER/SPACE to keep it, draw the next one, press ESC when finished.
# Both OpenCV inpainting algorithms (Telea and Navier-Stokes) are compared.
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="church.jpeg")
parser.add_argument("--radius", type=int, default=3, help="inpaint neighbourhood radius")
parser.add_argument("--auto", action="store_true")
args = parser.parse_args()

img = cv2.imread(args.image)
if args.auto:
    # the TV tower and a cloud in church.jpeg (x, y, w, h)
    rois = [(708, 140, 34, 212), (528, 70, 162, 44)]
else:
    rois = cv2.selectROIs("Select ROIs (ENTER = keep, ESC = done)", img)
    cv2.destroyAllWindows()

mask = np.zeros(img.shape[:2], dtype="uint8")
for (x, y, w, h) in rois:
    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, thickness=-1)

telea = cv2.inpaint(img, mask, args.radius, cv2.INPAINT_TELEA)
ns = cv2.inpaint(img, mask, args.radius, cv2.INPAINT_NS)
cv2.imwrite('../Output/exercise 5.18 telea.jpg', telea)
cv2.imwrite('../Output/exercise 5.18 ns.jpg', ns)

marked = img.copy()
for (x, y, w, h) in rois:
    cv2.rectangle(marked, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

titles = ["Original + ROIs", "Mask", "Telea", "Navier-Stokes"]
images = [marked, mask, telea, ns]
plt.figure(figsize=(14, 9))
for i, (t, im) in enumerate(zip(titles, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title(t)
    plt.axis('off')
plt.tight_layout()
plt.savefig('../Output/exercise 5.18.png')
plt.show()
