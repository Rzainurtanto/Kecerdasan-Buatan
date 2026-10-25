# Example 5.10a StreamlitWebcamImageClassification.py
# Run: streamlit run example5.10a.py
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# option: (label, keras module, constructor, image_size)
MODELS = {
    "VGG16": ("VGG16", vgg16, vgg16.VGG16, 224),
    "RESNET50": ("ResNet50", resnet50, resnet50.ResNet50, 224),
    "MOBILENET": ("MobileNet", mobilenet, mobilenet.MobileNet, 224),
    "INCEPTION_V3": ("Inception", inception_v3, inception_v3.InceptionV3, 299),
}


@st.cache_resource
def get_cap():
    return cv2.VideoCapture(0)


@st.cache_resource
def load_model(option):
    return MODELS[option][2](weights='imagenet')


def model_predict(cam_frame, option):
    label_name, module, _, image_size = MODELS[option]
    frame = cv2.resize(cam_frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # webcam frames are BGR, the models expect RGB
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = module.preprocess_input(image_batch.copy())
    predictions = load_model(option).predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    cv2.putText(cam_frame, "{}: {}, {:.2f}".format(label_name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return cam_frame


cap = get_cap()

frameST = st.empty()
st.title("Image Classification")
st.sidebar.markdown("# Image Classification")
option = st.sidebar.selectbox(
    'Select a Deep Learning Model:',
    list(MODELS), index=0)
st.sidebar.write('You selected:', option)

while True:
    ret, frame = cap.read()
    # Stop the program if reached end of video
    if not ret:
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    frame = model_predict(frame, option)
    frameST.image(frame, channels="BGR")
