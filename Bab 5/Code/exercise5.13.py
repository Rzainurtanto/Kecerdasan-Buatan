# Exercise 5.13 Example 5.10 reading images from a video file instead of the webcam
# Run: streamlit run exercise5.13.py
# Pick a video in the sidebar (sample_video.mp4 is provided) or upload your own.
import os
import tempfile
import time
import streamlit as st
import cv2
import numpy as np

st.title("Video Player")
uploaded = st.sidebar.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])
contrast = st.sidebar.slider('Contrast', 0, 100, 50) / 50.0
brightness = st.sidebar.slider('Brightness', 0, 100, 0)
loop = st.sidebar.checkbox('Loop', value=True)

if uploaded is not None:
    # OpenCV needs a real file name, so the upload is written to a temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read())
    tmp.close()
    video_file = tmp.name
else:
    video_file = 'sample_video.mp4'
st.sidebar.write('Playing:', uploaded.name if uploaded else video_file)

# cv2.VideoCapture(0) -> cv2.VideoCapture(<video file name>)
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
frameST = st.empty()

while True:
    ret, frame = cap.read()
    # Stop (or restart) when the end of the video is reached
    if not ret:
        if loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        print("Done processing !!!")
        cap.release()
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:, :, 2] = np.clip(contrast * frame[:, :, 2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    frameST.image(frame, channels="BGR")
    time.sleep(1 / fps)
