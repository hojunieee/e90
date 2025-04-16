import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {i} opened, frame: {'OK' if ret else 'Fail'}")
        cap.release()
