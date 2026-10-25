# Example 5.4 Webcam.py
import cv2

id = 0
camera = cv2.VideoCapture(id)
# image_size = 224

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    # cam_frame = cv2.resize(cam_frame, (image_size, image_size))
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
