# Exercise 5.22 Fractalius effect with six orientations (0, pi/6, 2pi/6, 3pi/6, 4pi/6, 5pi/6)
# Shown next to the original and the four-orientation version from Example 5.16.
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fractalius(image, No, j=1):
    filtered = np.zeros(image.shape, np.float32)
    for i in range(No):  # Orientation
        kernel = cv2.getGaborKernel((21, 21), 4, np.pi / No * i, 4 * (j + 1), 1, 0, cv2.CV_32F)
        kernel /= math.sqrt((kernel * kernel).sum())
        fimg = cv2.filter2D(image, -1, kernel)
        filtered = np.maximum(filtered, fimg)
    return np.clip(filtered, 0, 1)


files = ['pineapple.jpeg', 'bird2.jpeg']
plt.figure(figsize=(15, 5 * len(files)))
for r, f in enumerate(files):
    image = cv2.imread(f).astype(np.float32) / 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for c, (title, im) in enumerate((("Original Image", image),
                                     ("4 orientations", fractalius(image, 4)),
                                     ("6 orientations", fractalius(image, 6)))):
        plt.subplot(len(files), 3, r * 3 + c + 1)
        plt.imshow(im)
        plt.axis('off')
        plt.title("{} - {}".format(f, title))
plt.tight_layout()
plt.savefig('../Output/exercise 5.22.png')
plt.show()
