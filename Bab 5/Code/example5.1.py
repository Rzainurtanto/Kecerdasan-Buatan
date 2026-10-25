# Example 5.1 KerasModel.py
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet_v2,
    inception_v3,
    efficientnet
)

# init the models
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet_v2.MobileNetV2(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')
model = efficientnet.EfficientNetB0(weights='imagenet')
model.summary()
