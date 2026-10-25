# Example 5.9a StreamlitImageClassification.py
# https://github.com/streamlit/streamlit/issues/511
# Run: streamlit run example5.9a.py
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
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


# each model is loaded once and kept, so switching back and forth is instant
@st.cache_resource
def load_model(option):
    return MODELS[option][2](weights='imagenet')


def model_predict(cam_frame, option):
    label_name, module, _, image_size = MODELS[option]
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    # prepare the image for the selected model
    processed_image = module.preprocess_input(image_batch.copy())
    predictions = load_model(option).predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    cv2.putText(cam_frame, "{}: {}, {:.2f}".format(label_name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cam_frame


frameST = st.empty()
st.title("Image Classification")
st.sidebar.markdown("# Image Classification")
option = st.sidebar.selectbox(
    'Select a deep learning Model:',
    list(MODELS), index=0)
st.sidebar.write('You selected:', option)

file_image = st.sidebar.file_uploader("Upload your Images", type=['jpeg', 'jpg', 'png', 'gif'])

if file_image is None:
    st.write("No image file!")
else:
    img = np.array(Image.open(file_image).convert('RGB'))
    st.write("Image")

    img = model_predict(img, option)
    st.image(img, width="stretch")

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG')
    st.download_button("Download", buf.getvalue(), file_name='output.jpg', mime='image/jpeg')
