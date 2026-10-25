# Example 5.14 GaborFilter.py
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sinusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

No = 4
Nw = 4
N = No * Nw
n = 0
plt.figure(figsize=(10, 10))
for j in range(Nw):  # wavelength
    for i in range(No):  # Orientation
        kernel = cv2.getGaborKernel((21, 21), 5, np.pi / 4 * i, 4 * (j + 1), 1, 0, cv2.CV_32F)
        kernel /= math.sqrt((kernel * kernel).sum())

        n = n + 1
        plt.subplot(Nw, No, n)
        plt.imshow(kernel, cmap='rainbow')
plt.savefig('../Output/example 5.14.png')
plt.show()
