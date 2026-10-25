# Exercise 5.19 Colour old black-and-white photos
# PySimpleGUI-Photo-Colorizer (https://github.com/PySimpleGUI/PySimpleGUI-Photo-Colorizer) wraps the
# Zhang et al. "Colorful Image Colorization" Caffe model in a PySimpleGUI window. PySimpleGUI 5 now
# needs a paid licence key, so this script runs the same model with OpenCV DNN directly.
# Model files (downloaded into models/): colorization_deploy_v2.prototxt,
# colorization_release_v2.caffemodel, pts_in_hull.npy
# Usage: python exercise5.19.py                       (demo: a colour photo turned grey, then recoloured)
#        python exercise5.19.py old_photo.jpg         (colour your own black-and-white photo)
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

prototxt = os.path.join('models', 'colorization_deploy_v2.prototxt')
model = os.path.join('models', 'colorization_release_v2.caffemodel')
points = os.path.join('models', 'pts_in_hull.npy')

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)
# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


def colorize(image_bgr):
    """Predict the a/b colour channels from the L (lightness) channel and merge them back."""
    scaled = image_bgr.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # mean centering
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image_bgr.shape[1], image_bgr.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")


photos = sys.argv[1:] or ['church.jpeg', 'fruits.jpg', 'messi5.jpg']

fig = plt.figure(figsize=(14, 4 * len(photos)))
for r, name in enumerate(photos):
    original = cv2.imread(name)
    gray = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = colorize(gray)
    cv2.imwrite('../Output/exercise 5.19 {}'.format(os.path.basename(name)), result)
    for c, (title, im) in enumerate((("Original", original), ("Black & white", gray), ("Colorized", result))):
        ax = fig.add_subplot(len(photos), 3, r * 3 + c + 1)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title("{} - {}".format(name, title))
        ax.axis('off')
plt.tight_layout()
plt.savefig('../Output/exercise 5.19.png')
plt.show()
