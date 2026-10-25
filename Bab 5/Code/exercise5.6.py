# Exercise 5.6 Add more base models to the transfer learning program and compare them
# (models that are not in the book's list: VGG19, DenseNet121, Xception, EfficientNetB0,
#  MobileNetV3Large and ConvNeXtTiny). Same setup as exercise5.5.py.
from transfer_utils import compare_models

base_models = ["VGG19", "DenseNet121", "Xception", "EfficientNetB0", "MobileNetV3Large", "ConvNeXtTiny"]

compare_models(base_models, "Flowers", epochs=10,
               title="Exercise 5.6 - additional base models on Flowers",
               save_as="../Output/exercise 5.6.png")
