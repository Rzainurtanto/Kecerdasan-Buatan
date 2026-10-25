# Exercise 5.5 Run the transfer learning program with different pre-trained models
# and compare the training time and the accuracy that can be achieved.
# Every model gets the same frozen base + GlobalAveragePooling + Dense(256) head,
# the same data split (Flowers, 5 classes) and the same number of epochs.
from transfer_utils import compare_models

base_models = ["VGG16", "ResNet50", "MobileNetV2", "DenseNet201", "InceptionV3"]

compare_models(base_models, "Flowers", epochs=10,
               title="Exercise 5.5 - book base models on Flowers",
               save_as="../Output/exercise 5.5.png")
