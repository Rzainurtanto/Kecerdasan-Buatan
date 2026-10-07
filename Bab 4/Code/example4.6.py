# Example 4.6 Incept.py
from tensorflow.keras.applications import inception_v3

# init the models
model = inception_v3.InceptionV3(weights='imagenet')
print(model.summary())
