# Exercise 4.11 Webcam Image Classification with VGG19
import time
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
import numpy as np

image_size = 224
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg19.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    start = time.time()
    predictions = model.predict(processed_image, verbose=0)
    predict_time = time.time() - start
    label = decode_predictions(predictions)

    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "VGG19: {}, {:.1f}".format(label[0][0][1], label[0][0][2]), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(cam_frame, "Time: {:.3f} s".format(predict_time), (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
