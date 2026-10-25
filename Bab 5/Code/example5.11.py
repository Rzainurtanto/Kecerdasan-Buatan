# Example 5.11 ImageStitching.py
# The book stitches London_left.jpg + London_right.jpg (Pexels). Those photos are not
# downloadable here, so this uses the OpenCV stitching test photos boat1/2.jpg.
import numpy as np
import cv2 as cv
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)
names = ['boat1.jpg', 'boat2.jpg']
imgs = []
for img_name in names:
    img = cv.imread(cv.samples.findFile(img_name))
    if img is None:
        print("can't read image " + img_name)
        sys.exit(-1)
    cv.imshow(img_name, img)
    imgs.append(img)

stitcher = cv.Stitcher.create(modes[0])
status, pano = stitcher.stitch(imgs)

if status != cv.Stitcher_OK:
    print("Can't stitch images, error code = %d" % status)
    sys.exit(-1)

cv.imwrite('../Output/example 5.11.jpg', pano)
cv.imshow('result', pano)
cv.waitKey(0)
cv.destroyAllWindows()
