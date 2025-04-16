import cv2
import apriltag
import numpy as np
import csv
import os
import time

# ==================== INPUT INTRINSIC AND EXTRINSIC PARAMETERS ==================== #
save_folder = "../E90/data/calibration/"
camera_param = os.path.join(save_folder, "stereo_calibration.npz")

if os.path.exists(camera_param):
    data = np.load(camera_param)
    camera_matrix1 = data['cam_matrix1']
    dist_coeffs1 = data['dist_coeffs1']
    camera_matrix2 = data['cam_matrix2']
    dist_coeffs2 = data['dist_coeffs2']
    R_2_1 = data['R_2_1']
    T_2_1 = data['T_2_1']
else:
    raise FileNotFoundError(f"Calibration file not found at {camera_param}")

# ==================== APRILTAG PARAMETERS ==================== #
tagsize = 0.091  # AprilTag physical size (m)
o_points = np.array([[-tagsize/2, tagsize/2, 0], 
                     [tagsize/2, tagsize/2, 0], 
                     [tagsize/2, -tagsize/2, 0], 
                     [-tagsize/2, -tagsize/2, 0]])

# ================= CAMERA SETUP ================= #
detector = apriltag.Detector()
cap1 = cv2.VideoCapture(1)  # Cam 1 (Middle)
cap2 = cv2.VideoCapture(2)  # Cam 2 (Right) - world frame

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()
# ================= CSV LIST ================= #
save_folder = "../E90/data/"
absolute_path = os.path.abspath(save_folder)
tag_estimation_path = os.path.join(absolute_path, "stereo_tag_estimations.csv")
csv_data = []  # [Time step, x1_new, y1_new, z1_new, x2, y2, z2]

# ================= APRILTAG DETECTION FUNCTION ================= #

def detect_tags(frame, camera_name, camera_matrix, dist_coeffs = None):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    results = []

    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners  # (x, y) coordinates of the four corners in the camera frame

        # Use solvePnP to estimate pose to get 3D position
        _, rvec, tvec = cv2.solvePnP(o_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        # rmat = cv2.Rodrigues(rvec)[0] @ np.diag([1, -1, -1]) # Get rotation matrix
        # pose = np.vstack((np.hstack((rmat, tvec)), [0, 0, 0, 1])) # 4x4 pose matrix
        # print(f"Pose Matrix for camera {camera_name}:\n{pose}\n")
        
        # Extract estimated position
        x, y, z = tvec.flatten() 
        results.append((x, y, z))  # Store estimated 3D position in respect to camera

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
cam1_list = []
cam2_list = []
frame_indices = []  # Keep track of frames where detection happened in both cameras

frame_idx = 0  # Frame counter
prev_time = time.time()  # For FPS calculation
print("Starting dual camera feed...")
fps = []
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Failed to capture frames.")
        break

    # Detect tags using the correct camera parameters
    cam1_detected = detect_tags(frame1, "Cam 1", camera_matrix1)
    cam2_detected = detect_tags(frame2, "Cam 2", camera_matrix2)

    if cam1_detected and cam2_detected:  # Ensure both cameras detected a tag
        cam1_list.append(cam1_detected)
        cam2_list.append(cam2_detected)
        frame_indices.append(frame_idx)  # Store the frame number where both detected

    current_time = time.time()
    fps.append(1.0 / (current_time - prev_time))  # FPS calculation
    prev_time = current_time

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

    frame_idx += 1  # Increment frame counter

cap1.release()
cap2.release()
cv2.destroyAllWindows()

print("########################################")
print(f"Average FPS: {np.mean(fps):.2f}")
print(f"Total frames processed: {frame_idx}")
print(f"Frames with valid detections in both cameras: {len(frame_indices)}")
print("########################################")

# ================= PROCESS DATA (Use extrinsic) ================= #
csv_data = []

for i in range(len(cam1_list)):  # Iterate only over valid detections
    if not cam1_list[i] or not cam2_list[i]:  
        continue  # Skip frames where detection was missing

    # Extract detected tag positions from both cameras
    x1, y1, z1 = cam1_list[i][0] # Assuming only one tag detected per frame
    x2, y2, z2 = cam2_list[i][0] 

    # Convert to homogeneous coordinates
    p1 = np.array([x1, y1, z1]).reshape(3, 1)  # 

    # Apply extrinsic parameters (Rotation & Translation)
    p1_world = np.linalg.inv(R_2_1) @ (p1 - T_2_1)
    p1_new = p1_world.flatten()  # Flatten to 1D array

    # Store the results
    csv_data.append([frame_indices[i], p1_new[0], p1_new[1], p1_new[2], x1, y1, z1, x2, y2, z2])

# ================= SAVE DATA TO CSV (ONCE) ================= #
csv_filename = tag_estimation_path
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Frame", "x_cam1_new", "y_cam1_new", "z_cam1_new", "x_cam1", "y_cam1", "z_cam1", "x_cam2", "y_cam2", "z_cam2"])  # Header
    writer.writerows(csv_data)  # Write all collected data at once

print(f"Saved stereo tag estimations to {csv_filename}")
