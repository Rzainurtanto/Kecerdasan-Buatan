# Exercise 4.14 Select the Classification Model with an If-Else Statement
# https://pypi.org/project/image-classifiers/
# pip install image-classifiers tf_keras
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # image-classifiers needs Keras 2 (tf_keras), set before importing tensorflow
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers

import numpy as np
import cv2

print("Select a model:")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")
choice = input("Enter your choice (1-5): ")

# Set up a model
if choice == '1':
    name, sz = 'vgg16', 224
elif choice == '2':
    name, sz = 'resnet50', 224
elif choice == '3':
    name, sz = 'mobilenetv2', 224
elif choice == '4':
    name, sz = 'densenet201', 224
elif choice == '5':
    name, sz = 'inceptionv3', 299
else:
    print("Invalid choice, using VGG16")
    name, sz = 'vgg16', 224

clf, preprocess_input = Classifiers.get(name)
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
model.summary()

camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(cam_frame, (sz, sz))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
