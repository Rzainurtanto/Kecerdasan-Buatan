# Exercise 5.3 Identify several animal pictures with different models and compare them
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# animal pictures downloaded from Wikimedia Commons / OpenCV / book (Elephant.jpg)
images = ['Elephant.jpg', 'cat.jpg', 'bird2.jpeg', 'baboon.jpg']

models = {
    "VGG16": (vgg16.VGG16, vgg16.preprocess_input, 224),
    "ResNet50": (resnet50.ResNet50, resnet50.preprocess_input, 224),
    "MobileNet": (mobilenet.MobileNet, mobilenet.preprocess_input, 224),
    "InceptionV3": (inception_v3.InceptionV3, inception_v3.preprocess_input, 299),
}

rows = []
for name, (constructor, preprocess, size) in models.items():
    K.clear_session()
    model = constructor(weights='imagenet')
    for img_path in images:
        x = img_to_array(load_img(img_path, target_size=(size, size)))
        x = preprocess(np.expand_dims(x, axis=0))
        start = time.time()
        predictions = model.predict(x, verbose=0)
        elapsed = time.time() - start
        top = decode_predictions(predictions, top=1)[0][0]
        rows.append({"image": img_path, "model": name, "label": top[1],
                     "confidence": round(float(top[2]), 3), "time_s": round(elapsed, 3)})
        print("{:<12} {:<12} {:<24} {:.3f}  ({:.2f}s)".format(name, img_path, top[1], top[2], elapsed))

df = pd.DataFrame(rows)
print("\nTop-1 label per model:")
print(df.pivot(index="image", columns="model", values="label").to_string())
print("\nConfidence per model:")
print(df.pivot(index="image", columns="model", values="confidence").to_string())
print("\nAverage prediction time (s):")
print(df.groupby("model")["time_s"].mean().round(3).to_string())
df.to_csv("../Output/exercise 5.3.csv", index=False)

# show every image with each model's answer
fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 5))
for ax, img_path in zip(axes, images):
    ax.imshow(load_img(img_path))
    ax.axis('off')
    sub = df[df.image == img_path]
    ax.set_title("\n".join("{}: {} ({:.2f})".format(r.model, r.label, r.confidence)
                           for r in sub.itertuples()), fontsize=9)
plt.tight_layout()
plt.savefig('../Output/exercise 5.3.png')
plt.show()
