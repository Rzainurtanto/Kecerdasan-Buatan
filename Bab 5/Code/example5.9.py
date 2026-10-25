# Example 5.9 StreamlitImageClassification0.py
# pip install streamlit
# Run: streamlit run example5.9.py
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
from tensorflow.keras.applications import vgg16


# st.cache from the book was removed from Streamlit; cache_resource loads the model only once
@st.cache_resource
def load_model():
    return vgg16.VGG16(weights='imagenet')


def vgg16_predict(cam_frame, image_size):
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image, verbose=0)
    label_vgg = decode_predictions(predictions)
    cv2.putText(cam_frame, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cam_frame


model = load_model()
image_size = 224

frameST = st.empty()
st.title("Image Classification")
st.sidebar.markdown("# Image Classification")

file_image = st.sidebar.file_uploader("Upload your Images", type=['jpeg', 'jpg', 'png', 'gif'])
if file_image is None:
    st.write("No image file!")
else:
    # keep the image in RGB, the channel order Keras models and st.image expect
    img = np.array(Image.open(file_image).convert('RGB'))
    st.write("Image")

    img = vgg16_predict(img, image_size)
    st.image(img, width="stretch")

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG')
    st.download_button("Download", buf.getvalue(), file_name='output.jpg', mime='image/jpeg')
