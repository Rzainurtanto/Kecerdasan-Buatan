# Example 5.15 GaborFilter2.py
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pineapple.jpeg', 0).astype(np.float32) / 255
cv2.imshow('Image', image)

No = 4
Nw = 4
N = No * Nw
n = 0

plt.figure(figsize=(12, 10))
for j in range(Nw):  # wavelength
    for i in range(No):  # Orientation
        kernel = cv2.getGaborKernel((21, 21), 5, np.pi / 4 * i, 4 * (j + 1), 1, 0, cv2.CV_32F)
        kernel /= math.sqrt((kernel * kernel).sum())
        filtered = cv2.filter2D(image, -1, kernel)
        n = n + 1
        plt.subplot(Nw, No, n)
        plt.imshow(filtered, cmap='rainbow')
        plt.axis('off')
plt.savefig('../Output/example 5.15.png')
plt.show()
cv2.destroyAllWindows()
