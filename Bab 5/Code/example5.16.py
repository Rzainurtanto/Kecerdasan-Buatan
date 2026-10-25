# Example 5.16 GaborFilter3.py (Fractalius effect)
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('bird2.jpeg').astype(np.float32) / 255
image = cv2.imread('pineapple.jpeg').astype(np.float32) / 255
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
No = 4
j = 1
size = image.shape  # h, w, c = im.shape
# the book uses np.int8 here, which breaks np.maximum with float images
filtered = np.zeros(size, np.float32)

for i in range(No):  # Orientation
    kernel = cv2.getGaborKernel((21, 21), 4, np.pi / 4 * i, 4 * (j + 1), 1, 0, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    fimg = cv2.filter2D(image, -1, kernel)
    filtered = np.maximum(filtered, fimg)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(np.clip(filtered, 0, 1))
plt.axis('off')
plt.title('Gabor Filtered Image')
plt.savefig('../Output/example 5.16.png')
plt.show()
