# Exercise 4.10 Image Classification with VGG19 Compared to VGG16
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

models = {
    'VGG16': (vgg16.VGG16(weights='imagenet'), vgg16.preprocess_input),
    'VGG19': (vgg19.VGG19(weights='imagenet'), vgg19.preprocess_input),
}

for name, (model, preprocess_input) in models.items():
    processed_image = preprocess_input(x.copy())
    model.predict(processed_image, verbose=0)  # warm up

    # prediction
    start = time.time()
    predictions = model.predict(processed_image, verbose=0)
    predict_time = time.time() - start
    results = decode_predictions(predictions)

    print("\n%s (params: %s, prediction time: %.3f s)" % (name, format(model.count_params(), ','), predict_time))
    print(results)
