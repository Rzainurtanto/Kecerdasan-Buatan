# Exercise 5.20 Image super resolution with different deep learning models
# Instead of the webcam, a known image is shrunk by the model's scale and enlarged again,
# so every model can be compared with the real original using PSNR (higher = closer)
# and its running time. Bicubic interpolation is the non-deep-learning baseline.
# Models (in models/): EDSR_x4, ESPCN_x4, FSRCNN_x3, LapSRN_x8
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('baboon.jpg')  # 512 x 512, lots of fine texture
model_files = ['EDSR_x4.pb', 'ESPCN_x4.pb', 'FSRCNN_x3.pb', 'LapSRN_x8.pb']

panels = []
print("{:<10}{:>6}{:>12}{:>14}{:>10}".format("model", "scale", "PSNR (dB)", "bicubic PSNR", "time (s)"))
for modelfile in model_files:
    modelName = modelfile.split("_")[0].lower()
    modelScale = int(modelfile.split("_x")[-1].split(".")[0])

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(os.path.join('models', modelfile))
    sr.setModel(modelName, modelScale)

    # crop so the size divides by the scale, then shrink it
    h, w = (image.shape[0] // modelScale) * modelScale, (image.shape[1] // modelScale) * modelScale
    original = image[:h, :w]
    small = cv2.resize(original, (w // modelScale, h // modelScale), interpolation=cv2.INTER_AREA)

    start = time.time()
    upscaled = sr.upsample(small)
    elapsed = time.time() - start
    bicubic = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    psnr_sr = cv2.PSNR(original, upscaled)
    psnr_bic = cv2.PSNR(original, bicubic)
    print("{:<10}{:>6}{:>12.2f}{:>14.2f}{:>10.2f}".format(modelName.upper(), modelScale, psnr_sr, psnr_bic, elapsed))
    cv2.imwrite('../Output/exercise 5.20 {}.png'.format(modelName), upscaled)
    panels.append((modelName.upper(), modelScale, small, bicubic, upscaled, psnr_bic, psnr_sr, elapsed))

# zoom into the same 128x128 area (the eye) of every result
y0, x0, s = 60, 100, 128
fig, axes = plt.subplots(len(panels), 3, figsize=(10, 3.4 * len(panels)))
for row, (name, scale, small, bic, up, pb, ps, t) in zip(axes, panels):
    row[0].imshow(cv2.cvtColor(cv2.resize(small, (bic.shape[1], bic.shape[0]), interpolation=cv2.INTER_NEAREST)[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB))
    row[0].set_title("input (1/{} size)".format(scale))
    row[1].imshow(cv2.cvtColor(bic[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB))
    row[1].set_title("bicubic x{}  {:.2f} dB".format(scale, pb))
    row[2].imshow(cv2.cvtColor(up[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB))
    row[2].set_title("{} x{}  {:.2f} dB  {:.2f}s".format(name, scale, ps, t))
    for a in row:
        a.axis('off')
plt.tight_layout()
plt.savefig('../Output/exercise 5.20.png')
plt.show()
