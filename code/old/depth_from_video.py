import cv2
import apriltag
import numpy as np

# Camera parameters (from calibration)
fx = 1439.8957446731756  # Focal length X
fy = 1440.8478532021038  # Focal length Y
cx = 940.985979  # Principal point X (from calibration)
cy = 555.600242  # Principal point Y (from calibration)
tagsize = 0.091  # AprilTag physical size in meters

# Camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

# Define AprilTag detector
detector = apriltag.Detector()

cap1 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

if not cap1.isOpened():
    print("Error: Could not open camera.")
    exit()

o_points = np.array([[-tagsize/2, tagsize/2, 0], [tagsize/2, tagsize/2, 0], [tagsize/2, -tagsize/2, 0], [-tagsize/2, -tagsize/2, 0]])


while True:
    ret, frame = cap1.read()
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

        # pose, _, _ = detector.detection_pose(detection, (fx, fy, cx, cy), tagsize)

        retval, rvec, tvec = cv2.solvePnP(o_points, corners, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        rmat = cv2.Rodrigues(rvec)[0] @ np.diag([1, -1, -1])

        pose2 = np.vstack((np.hstack((rmat, tvec)), [0, 0, 0, 1]))

        # dist = np.linalg.norm(pose[:3, 3])

        #print(f'retval={retval}, rvec={rvec}, tvec={tvec}')

        # Convert corner points to integers for OpenCV
        corners = [(int(p[0]), int(p[1])) for p in corners]

        # Draw a red box around the tag
        cv2.line(frame, corners[0], corners[1], (0, 0, 255), 2)
        cv2.line(frame, corners[1], corners[2], (0, 0, 255), 2)
        cv2.line(frame, corners[2], corners[3], (0, 0, 255), 2)
        cv2.line(frame, corners[3], corners[0], (0, 0, 255), 2)

        # Put the tag ID label near the top-left corner of the tag
        cv2.putText(
            frame, f"ID: {tag_id}, dist={pose2}", (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

        print(f"Detected tag ID: {tag_id} at {detection.center}")
        # print(f"Pose:\n{pose}")
        print(f"Pose2:\n{pose2}")   

    # Show the frame with detected tags
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cv2.destroyAllWindows()
