# Exercise 4.13 Compare Different Models from the Image-Classifiers Library
# https://pypi.org/project/image-classifiers/
# pip install image-classifiers tf_keras
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # image-classifiers needs Keras 2 (tf_keras), set before importing tensorflow
import time
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers

import numpy as np
import cv2

# The models that were commented out in Example 4.15, with their input size
model_names = [('vgg16', 224), ('resnet50', 224), ('mobilenetv2', 224),
               ('densenet201', 224), ('inceptionv3', 299)]

frame_bgr = cv2.imread('Elephant.jpg')

results = []
for name, sz in model_names:
    # Set up a model
    clf, preprocess_input = Classifiers.get(name)
    model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)

    frame = cv2.resize(frame_bgr, (sz, sz))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    model.predict(image, verbose=0)  # warm up
    start = time.time()
    preds = model.predict(image, verbose=0)
    predict_time = time.time() - start
    label = decode_predictions(preds)

    results.append((name, model.count_params(), predict_time, label[0][0][1], label[0][0][2]))

print("\nModel       | Params      | Time (s) | Top-1 prediction")
for name, params, t, label, score in results:
    print("%-11s | %11s | %8.3f | %s (%.3f)" % (name, format(params, ','), t, label, score))
