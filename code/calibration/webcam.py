# This code runs the AprilTag detector on a live webcam (don't give us the depth data)

import cv2
import apriltag

img_width = 1280
img_height = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

detector = apriltag.Detector()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)

    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners  # (x, y) coordinates of the four corners

        # Convert corner points to integers for OpenCV
        corners = [(int(p[0]), int(p[1])) for p in corners]

        # Draw a red box around the tag
        cv2.line(frame, corners[0], corners[1], (0, 0, 255), 2)
        cv2.line(frame, corners[1], corners[2], (0, 0, 255), 2)
        cv2.line(frame, corners[2], corners[3], (0, 0, 255), 2)
        cv2.line(frame, corners[3], corners[0], (0, 0, 255), 2)

        # Put the tag ID label near the top-left corner of the tag
        cv2.putText(
            frame, f"ID: {tag_id}", (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

        print(f"Detected tag ID: {tag_id} at {detection.center}")

    # Show the frame with detected tags
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
