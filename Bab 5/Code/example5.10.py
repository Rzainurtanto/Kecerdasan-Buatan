# Example 5.10 StreamlitWebcam.py
# Run: streamlit run example5.10.py
import streamlit as st
import cv2
import numpy as np


@st.cache_resource
def get_cap():
    return cv2.VideoCapture(0)


cap = get_cap()

frameST = st.empty()
# the book's sliders start at 0, which turns the picture black; start at "no change" instead
contrast = st.sidebar.slider('Contrast', 0, 100, 50) / 50.0
brightness = st.sidebar.slider('Brightness', 0, 100, 0)

while True:
    ret, frame = cap.read()
    # Stop the program if reached end of video
    if not ret:
        print("Done processing !!!")
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:, :, 2] = np.clip(contrast * frame[:, :, 2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    frameST.image(frame, channels="BGR")
