# Example 5.2 KerasModel2.py
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# init the models
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')
model.summary()

img_path = 'Elephant.jpg'
img = load_img(img_path, target_size=(224, 224))  # InceptionV3 uses (299, 299)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
# processed_image = vgg16.preprocess_input(x)
# processed_image = resnet50.preprocess_input(x)
processed_image = mobilenet.preprocess_input(x)
# processed_image = inception_v3.preprocess_input(x)

# prediction (the book passes x here; the preprocessed image is what the model expects)
predictions = model.predict(processed_image)
results = decode_predictions(predictions)
print(results)
