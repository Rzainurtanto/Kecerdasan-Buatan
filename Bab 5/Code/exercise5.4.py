# Exercise 5.4 Example 5.5 with a selectable model and a webcam or video file as the source
# Usage: python exercise5.4.py --model mobilenet                    (webcam 0)
#        python exercise5.4.py --model vgg16 --source sample_video.mp4
# Press ESC to quit.
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import argparse
import time
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

MODELS = {
    "vgg16": (vgg16.VGG16, vgg16.preprocess_input, 224),
    "resnet50": (resnet50.ResNet50, resnet50.preprocess_input, 224),
    "mobilenet": (mobilenet.MobileNet, mobilenet.preprocess_input, 224),
    "inception_v3": (inception_v3.InceptionV3, inception_v3.preprocess_input, 299),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODELS, default="resnet50")
parser.add_argument("--source", default="0", help="webcam index (0, 1, ...) or a video file name")
parser.add_argument("--no-window", action="store_true", help="print the labels only")
args = parser.parse_args()

constructor, preprocess, image_size = MODELS[args.model]
model = constructor(weights='imagenet')

# replace 0 with the name of a video file to read from the file instead of the webcam
source = int(args.source) if args.source.isdigit() else args.source
camera = cv2.VideoCapture(source)

frames, total_time = 0, 0.0
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = preprocess(image_batch.copy())

    start = time.time()
    predictions = model.predict(processed_image, verbose=0)
    total_time += time.time() - start
    frames += 1
    label = decode_predictions(predictions)
    text = "{}: {}, {:.2f}".format(args.model, label[0][0][1], label[0][0][2])

    if args.no_window:
        print("frame {:>4}  {}".format(frames, text))
        continue
    cv2.putText(cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
if frames:
    print("{}: {} frames, average {:.3f}s per prediction ({:.1f} FPS)".format(
        args.model, frames, total_time / frames, frames / total_time))
