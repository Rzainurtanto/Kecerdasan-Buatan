# Example 5.12 ImageInpainting.py (Photo Inpaint Example)
# Drag a rectangle over the object to remove, then press ENTER or SPACE.
import numpy as np
import cv2

img = cv2.imread('church.jpeg')
rows, cols, channels = img.shape
print(rows, cols)
cv2.imshow('Original', img)

height, width, depth = img.shape
mask = np.zeros(shape=img.shape, dtype="uint8")
r = cv2.selectROI(img)
left = int(r[0])
top = int(r[1])
right = int(r[0] + r[2])
bottom = int(r[1] + r[3])

cv2.rectangle(img=mask,
              pt1=(left, top), pt2=(right, bottom),
              color=(255, 255, 255),
              thickness=-1)
# Apply the mask and display the result
maskedImg = cv2.bitwise_and(src1=img, src2=mask)
cv2.imshow("Masked", maskedImg)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
(thresh, mask) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Mask', mask)

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2.imwrite('../Output/example 5.12.jpg', dst)
cv2.imshow('Result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
