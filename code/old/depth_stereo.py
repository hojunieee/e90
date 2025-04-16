import cv2
import apriltag
import numpy as np
import csv
import os


# ================= CAMERA PARAMETERS ================= #
# Logitec Webcam parameters
fx1, fy1 = 1439.89, 1440.84  # Focal length (X, Y)
cx1, cy1 = 940.98, 555.60  # Camera center (X, Y)

camera_matrix1 = np.array([[fx1, 0, cx1],
                           [0, fy1, cy1],
                           [0,  0,  1]])

# My Mac Webcam parameters 
fx2, fy2 = 1351.67, 1354.66  # Focal length (X, Y)
cx2, cy2 = 961.39, 550.43  # Camera center (X, Y)

camera_matrix2 = np.array([[fx2, 0, cx2],
                           [0, fy2, cy2],
                           [0,  0,  1]])

tagsize = 0.091  # AprilTag physical size (m)
o_points = np.array([[-tagsize/2, tagsize/2, 0], 
                     [tagsize/2, tagsize/2, 0], 
                     [tagsize/2, -tagsize/2, 0], 
                     [-tagsize/2, -tagsize/2, 0]])

# ================= CAMERA SETUP ================= #
detector = apriltag.Detector()
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(2)  # Second camera

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# ================= CSV list ================= #
save_folder = "../E90/apriltag/test/data/"
absolute_path = os.path.abspath(save_folder)
tag_estimation_path = os.path.join(absolute_path, "tag_estimations.csv")
csv_data = []  # [x1, y1, z1, x2, y2, z2]

# ================= APRILTAG DETECTION FUNCTION ================= #

def detect_tags(frame, camera_name, camera_matrix):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    results = []

    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners  # (x, y) coordinates of the four corners

        # Use solvePnP to estimate pose instead of detection_pose

        _, rvec, tvec = cv2.solvePnP(o_points, corners, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rmat = cv2.Rodrigues(rvec)[0] @ np.diag([1, -1, -1])
        pose = np.vstack((np.hstack((rmat, tvec)), [0, 0, 0, 1]))
        
        # Extract estimated position
        x, y, z = tvec.flatten()
        results.append((x, y, z))  # Store estimated position

        # Convert corner points to integers for OpenCV
        corners = [(int(p[0]), int(p[1])) for p in corners]

        # Draw a red box around the tag
        cv2.line(frame, corners[0], corners[1], (0, 0, 255), 2)
        cv2.line(frame, corners[1], corners[2], (0, 0, 255), 2)
        cv2.line(frame, corners[2], corners[3], (0, 0, 255), 2)
        cv2.line(frame, corners[3], corners[0], (0, 0, 255), 2)

        # Display depth and print
        cv2.putText(frame, f"{camera_name} {tvec[2][0]:.2f}m", 
                    (corners[0][0], corners[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        # print(f"{camera_name} - Detected tag ID: {tag_id} at {detection.center}")
        # print(f"Pose Matrix:\n{pose}\n")

    return results


# ================= MAIN LOOP ================= #
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Failed to capture frames.")
        break

    # Detect tags using the correct camera parameters
    results_cam1 = detect_tags(frame1, "Logi", camera_matrix1)
    results_cam2 = detect_tags(frame2, "Mac", camera_matrix2)

    # Assuming one tag per frame from each camera
    if results_cam1 and results_cam2:
        (x1, y1, z1) = results_cam1[0]  # First detection from WebCam
        (x2, y2, z2) = results_cam2[0]  # First detection from Mac

        # Append to list
        csv_data.append([x1, y1, z1, x2, y2, z2])

    # Some resizing so that we can display side by side
    height = min(frame1.shape[0], frame2.shape[0])
    width = min(frame1.shape[1], frame2.shape[1])
    frame1_resized = cv2.resize(frame1, (width, height))
    frame2_resized = cv2.resize(frame2, (width, height))

    combined_frame = np.hstack((frame1_resized, frame2_resized))

    cv2.imshow("Dual Camera Feed", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= SAVE DATA TO CSV (ONCE) ================= #
csv_filename = tag_estimation_path
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x_cam1", "y_cam1", "z_cam1", "x_cam2", "y_cam2", "z_cam2"])  # Header
    writer.writerows(csv_data)  # Write all collected data at once

# ================= CLEANUP ================= #
cap1.release()
cap2.release()
cv2.destroyAllWindows()
