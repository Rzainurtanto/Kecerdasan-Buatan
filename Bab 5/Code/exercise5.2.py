# Exercise 5.2 Exercise 5.1 plus VGG19, DenseNet201 and EfficientNetB7
# Usage: python exercise5.2.py            (asks for a choice)
#        python exercise5.2.py 7          (shows DenseNet201 directly)
#        python exercise5.2.py all        (prints a size comparison of every model)
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import sys
from tensorflow.keras.applications import (
    vgg16,
    vgg19,
    resnet50,
    mobilenet_v2,
    inception_v3,
    densenet,
    efficientnet
)

print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. InceptionV3")
print("5. EfficientNetB0")
print("6. VGG19")
print("7. DenseNet201")
print("8. EfficientNetB7")
choice = sys.argv[1] if len(sys.argv) > 1 else input("Select a model (1-8 or 'all'): ").strip()


def create_model(choice):
    if choice == "1":
        return vgg16.VGG16(weights='imagenet')
    elif choice == "2":
        return resnet50.ResNet50(weights='imagenet')
    elif choice == "3":
        return mobilenet_v2.MobileNetV2(weights='imagenet')
    elif choice == "4":
        return inception_v3.InceptionV3(weights='imagenet')
    elif choice == "5":
        return efficientnet.EfficientNetB0(weights='imagenet')
    elif choice == "6":
        return vgg19.VGG19(weights='imagenet')
    elif choice == "7":
        return densenet.DenseNet201(weights='imagenet')
    elif choice == "8":
        return efficientnet.EfficientNetB7(weights='imagenet')  # ~256 MB download
    else:
        return None


if choice == "all":
    # weights=None builds the architecture only, so no weights need to be downloaded
    from tensorflow.keras import backend as K
    builders = {"1": vgg16.VGG16, "2": resnet50.ResNet50, "3": mobilenet_v2.MobileNetV2,
                "4": inception_v3.InceptionV3, "5": efficientnet.EfficientNetB0, "6": vgg19.VGG19,
                "7": densenet.DenseNet201, "8": efficientnet.EfficientNetB7}
    print("\n{:<18}{:>8}{:>16}".format("Model", "Layers", "Parameters"))
    for key, build in builders.items():
        K.clear_session()
        m = build(weights=None)
        print("{:<18}{:>8}{:>16,}".format(m.name, len(m.layers), m.count_params()))
    sys.exit(0)

model = create_model(choice)
if model is None:
    print("Unknown choice:", choice)
    sys.exit(1)
model.summary()
print("Model: {} | layers: {} | parameters: {:,}".format(model.name, len(model.layers), model.count_params()))
