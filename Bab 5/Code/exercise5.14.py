# Exercise 5.14 Example 5.10a with another deep learning model (VGG19) added
# Run: streamlit run exercise5.14.py
# The source can be the webcam or a video file (handy when there is no webcam).
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import time
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    vgg19,
    resnet50,
    mobilenet,
    inception_v3
)

# option: (label, keras module, constructor, image_size)
MODELS = {
    "VGG16": ("VGG16", vgg16, vgg16.VGG16, 224),
    "VGG19": ("VGG19", vgg19, vgg19.VGG19, 224),
    "RESNET50": ("ResNet50", resnet50, resnet50.ResNet50, 224),
    "MOBILENET": ("MobileNet", mobilenet, mobilenet.MobileNet, 224),
    "INCEPTION_V3": ("Inception", inception_v3, inception_v3.InceptionV3, 299),
}


@st.cache_resource
def load_model(option):
    return MODELS[option][2](weights='imagenet')


def model_predict(cam_frame, option):
    label_name, module, _, image_size = MODELS[option]
    frame = cv2.resize(cam_frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = module.preprocess_input(image_batch.copy())
    predictions = load_model(option).predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    cv2.putText(cam_frame, "{}: {}, {:.2f}".format(label_name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return cam_frame


st.title("Image Classification")
st.sidebar.markdown("# Image Classification")
option = st.sidebar.selectbox('Select a Deep Learning Model:', list(MODELS), index=1)
st.sidebar.write('You selected:', option)
source = st.sidebar.radio('Source', ['Webcam', 'sample_video.mp4'])
fps_text = st.sidebar.empty()

cap = cv2.VideoCapture(0 if source == 'Webcam' else source)
frameST = st.empty()
while True:
    ret, frame = cap.read()
    if not ret:
        if source != 'Webcam':  # loop the video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        st.write("No webcam frame")
        cap.release()
        break

    start = time.time()
    frame = model_predict(frame, option)
    fps_text.write("{:.1f} FPS".format(1 / (time.time() - start)))
    frameST.image(frame, channels="BGR")
