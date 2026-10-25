# Exercise 5.1 Select which model to display with an if-else structure
# Usage: python exercise5.1.py            (asks for a choice)
#        python exercise5.1.py 3          (shows MobileNetV2 directly)
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import sys
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet_v2,
    inception_v3,
    efficientnet
)

print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. InceptionV3")
print("5. EfficientNetB0")
choice = sys.argv[1] if len(sys.argv) > 1 else input("Select a model (1-5): ").strip()

# init the selected model
if choice == "1":
    model = vgg16.VGG16(weights='imagenet')
elif choice == "2":
    model = resnet50.ResNet50(weights='imagenet')
elif choice == "3":
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
elif choice == "4":
    model = inception_v3.InceptionV3(weights='imagenet')
elif choice == "5":
    model = efficientnet.EfficientNetB0(weights='imagenet')
else:
    print("Unknown choice:", choice)
    sys.exit(1)

model.summary()
print("Model: {} | layers: {} | parameters: {:,}".format(model.name, len(model.layers), model.count_params()))
