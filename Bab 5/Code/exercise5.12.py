# Exercise 5.12 Example 5.9a with another deep learning model (VGG19) added
# Run: streamlit run exercise5.12.py
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = module.preprocess_input(image_batch.copy())
    predictions = load_model(option).predict(processed_image, verbose=0)
    label = decode_predictions(predictions, top=3)
    cv2.putText(cam_frame, "{}: {}, {:.2f}".format(label_name, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cam_frame, label[0]


st.title("Image Classification")
st.sidebar.markdown("# Image Classification")
option = st.sidebar.selectbox('Select a deep learning Model:', list(MODELS), index=1)
st.sidebar.write('You selected:', option)

file_image = st.sidebar.file_uploader("Upload your Images", type=['jpeg', 'jpg', 'png', 'gif'])
if file_image is None:
    st.write("No image file!")
else:
    img = np.array(Image.open(file_image).convert('RGB'))
    img, top3 = model_predict(img, option)
    st.image(img, width="stretch")
    st.table({"label": [t[1] for t in top3], "probability": [round(float(t[2]), 3) for t in top3]})

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG')
    st.download_button("Download", buf.getvalue(), file_name='output.jpg', mime='image/jpeg')
