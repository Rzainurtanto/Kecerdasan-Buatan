# Exercise 5.21 Gabor filters with 6 wavelengths (4, 8, 12, 16, 20, 24)
# and 6 orientations (0, pi/6, 2pi/6, 3pi/6, 4pi/6, 5pi/6)
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

No = 6  # orientations
Nw = 6  # wavelengths
n = 0
plt.figure(figsize=(12, 12))
for j in range(Nw):  # wavelength
    for i in range(No):  # Orientation
        theta = np.pi / No * i
        lambd = 4 * (j + 1)
        # a larger kernel (31x31) so the long wavelengths (20, 24) still fit inside it
        kernel = cv2.getGaborKernel((31, 31), 5, theta, lambd, 1, 0, cv2.CV_32F)
        kernel /= math.sqrt((kernel * kernel).sum())

        n = n + 1
        plt.subplot(Nw, No, n)
        plt.imshow(kernel, cmap='rainbow')
        plt.axis('off')
        if j == 0:
            plt.title("theta = {}pi/6".format(i))
        if i == 0:
            plt.text(-12, 15, "lambda={}".format(lambd), rotation=90, va='center')
plt.suptitle("Gabor kernels: rows = wavelength, columns = orientation")
plt.savefig('../Output/exercise 5.21.png')
plt.show()
