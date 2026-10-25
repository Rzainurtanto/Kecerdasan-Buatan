# Example 5.3 WebcamScan.py
import sys
import cv2

# Hide OpenCV's internal "Camera index out of range" messages for indices without a camera
cv2.setLogLevel(0)  # 0 = LOG_LEVEL_SILENT
# DirectShow is the native Windows camera backend: faster to probe than the default one
backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

found = []
for i in range(5):
    print("Checking camera #{0}...".format(i))
    cap = cv2.VideoCapture(i, backend)
    if cap.isOpened():
        print("Camera found!")
        found.append(i)
    if not cap.isOpened():
        print("Camera not found!")
    cap.release()

# "Camera not found!" only means there is no camera at that index, it is not an error
print("\n{} camera(s) available: {}".format(len(found), ", ".join("#{}".format(i) for i in found) or "none"))
if found:
    print("Use cv2.VideoCapture({}) in the webcam examples.".format(found[0]))
