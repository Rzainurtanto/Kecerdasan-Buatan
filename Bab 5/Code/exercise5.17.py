# Exercise 5.17 Stitch three images together
# Also compares the two stitcher modes (PANORAMA for rotating cameras, SCANS for flat scans).
import cv2 as cv
import sys
import matplotlib.pyplot as plt

names = ['boat1.jpg', 'boat2.jpg', 'boat3.jpg']
imgs = []
for img_name in names:
    img = cv.imread(img_name)
    if img is None:
        print("can't read image " + img_name)
        sys.exit(-1)
    imgs.append(img)

results = {}
for mode_name, mode in (("PANORAMA", cv.Stitcher_PANORAMA), ("SCANS", cv.Stitcher_SCANS)):
    stitcher = cv.Stitcher.create(mode)
    status, pano = stitcher.stitch(imgs)
    if status != cv.Stitcher_OK:
        print("{}: can't stitch images, error code = {}".format(mode_name, status))
        continue
    print("{}: stitched {} images -> {} x {}".format(mode_name, len(imgs), pano.shape[1], pano.shape[0]))
    results[mode_name] = pano
    cv.imwrite('../Output/exercise 5.17 {}.jpg'.format(mode_name.lower()), pano)

# inputs on the first row, one stitched result per row below
fig = plt.figure(figsize=(15, 4 + 4 * len(results)))
for i, img in enumerate(imgs):
    ax = fig.add_subplot(1 + len(results), 3, i + 1)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax.set_title(names[i])
    ax.axis('off')
for r, (mode_name, pano) in enumerate(results.items()):
    ax = fig.add_subplot(1 + len(results), 1, r + 2)
    ax.imshow(cv.cvtColor(pano, cv.COLOR_BGR2RGB))
    ax.set_title("Stitched result ({})".format(mode_name))
    ax.axis('off')
plt.tight_layout()
plt.savefig('../Output/exercise 5.17.png')
plt.show()
