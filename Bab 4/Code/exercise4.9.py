# Exercise 4.9 Load the VGG19 Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

model = VGG19(weights='imagenet')
print(model.summary())
