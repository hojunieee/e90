import cv2
import apriltag
import numpy as np
import csv
import os
import time

# ==================== INPUT INTRINSIC AND EXTRINSIC PARAMETERS ==================== #
save_folder = "../E90/data/calibration/"
camera_param = os.path.join(save_folder, "triple_calibration.npz")
frame_save_dir = os.path.join(save_folder, "triple_tag_frames")
os.makedirs(frame_save_dir, exist_ok=True)

if os.path.exists(camera_param):
    data = np.load(camera_param)
    K0 = data['K_0']
    K1 = data['K_1']
    K2 = data['K_2']
    dist_coeffs0 = data['dist_0']
    dist_coeffs1 = data['dist_1']
    dist_coeffs2 = data['dist_2']
    R_0 = data['R_0']
    R_2 = data['R_2']
    T_0 = data['T_0']
    T_2 = data['T_2']
    E_0 = data['E_0']
    E_2 = data['E_2']
    F_0 = data['F_0']
    F_2 = data['F_2']
else:
    raise FileNotFoundError(f"Calibration file not found at {camera_param}")

# ==================== APRILTAG PARAMETERS ==================== #
tagsize = 0.09  # AprilTag physical size (m)
o_points = np.array([[-tagsize/2, tagsize/2, 0], 
                     [tagsize/2, tagsize/2, 0], 
                     [tagsize/2, -tagsize/2, 0], 
                     [-tagsize/2, -tagsize/2, 0]])

# ================= CAMERA SETUP ================= #
detector = apriltag.Detector()
cap0 = cv2.VideoCapture(0)  # Cam 0 (Left)
cap1 = cv2.VideoCapture(1)  # Cam 1 (Middle)
cap2 = cv2.VideoCapture(2)  # Cam 2 (Right) - world frame

if not cap1.isOpened() or not cap2.isOpened() or not cap0.isOpened():
    print("Error: Could not open at least one of the cameras.")
    exit()
# ================= CSV LIST ================= #
save_folder = "../E90/data/"
absolute_path = os.path.abspath(save_folder)
tag_estimation_path = os.path.join(absolute_path, "triple_tag_estimations.csv")
csv_data = []  

# ================= APRILTAG DETECTION FUNCTION ================= #

def detect_tags(frame, camera_name, camera_matrix, dist_coeffs=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    results = []

    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners  # 2D (pixel) corner positions

        # Pose estimation: tag frame to camera frame
        success, rvec, tvec = cv2.solvePnP(
            o_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue

        # Extract 3D pose: tag origin in camera frame
        position_3d = tvec.flatten()  # (x, y, z)
        center_2d = detection.center   # (u, v) in pixels

        results.append((position_3d, center_2d))

        # Draw box and label
        int_corners = [(int(p[0]), int(p[1])) for p in corners]
        for j in range(4):
            cv2.line(frame, int_corners[j], int_corners[(j+1)%4], (0, 0, 255), 2)

        cv2.putText(frame, f"{camera_name} {tvec[2][0]:.2f}m",
                    (int(center_2d[0]), int(center_2d[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    return results



# ================= MAIN LOOP ================= #
cam0_list = []
cam1_list = []
cam2_list = []
frame_indices = []  # Keep track of frames where detection happened in all cameras

frame_idx = 0  # Frame counter
fps = []

while True:
    prev_time = time.time()  # For FPS calculation
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2 or not ret0:
        print("Error: Failed to capture frames.")
        break

    # Detect tags using the correct camera parameters
    cam0_detected = detect_tags(frame0, "Cam 0", K0, None)
    cam1_detected = detect_tags(frame1, "Cam 1", K1, None)
    cam2_detected = detect_tags(frame2, "Cam 2", K2, None)

    if cam0_detected and cam1_detected and cam2_detected:  # Ensure all cameras detected a tag
        cam0_list.append(cam0_detected)
        cam1_list.append(cam1_detected)
        cam2_list.append(cam2_detected)
        frame_indices.append(frame_idx)  # Store the frame number where all detected
        cv2.imwrite(os.path.join(frame_save_dir, f"cam0_{frame_idx}.jpg"), frame0)
        cv2.imwrite(os.path.join(frame_save_dir, f"cam1_{frame_idx}.jpg"), frame1)
        cv2.imwrite(os.path.join(frame_save_dir, f"cam2_{frame_idx}.jpg"), frame2)

    current_time = time.time()
    fps.append(1.0 / (current_time - prev_time))  # FPS calculation

    # Some resizing so that we can display side by side
    height = min(frame0.shape[0], frame1.shape[0], frame2.shape[0])
    width = min(frame0.shape[1], frame1.shape[1], frame2.shape[1])
    frame0_resized = cv2.resize(frame0, (width, height))
    frame1_resized = cv2.resize(frame1, (width, height))
    frame2_resized = cv2.resize(frame2, (width, height))

    combined_frame = np.hstack((frame0_resized, frame1_resized, frame2_resized))

    cv2.imshow("Triple Camera Feed", combined_frame)

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
print(f"Frames with valid detections in all cameras: {len(frame_indices)}")
print("########################################")





# ================= PROCESS DATA (Using R, T + Undistortion) ================= #
csv_data = []

for i in range(len(cam1_list)):
    if not cam1_list[i] or not cam2_list[i] or not cam0_list[i]:
        continue

    # Unpack 3D pose and 2D pixel center
    cam0_pose, cam0_pixel = cam0_list[i][0]
    cam1_pose, cam1_pixel = cam1_list[i][0]
    cam2_pose, cam2_pixel = cam2_list[i][0]

    # Convert original tag position from camera frame to world (cam1) frame using R and T
    x_c0 = np.array(cam0_pose).reshape(3, 1)  # tag in cam0
    x_0_world = R_0 @ x_c0 + T_0              # convert to cam1 frame

    x_c2 = np.array(cam2_pose).reshape(3, 1)  # tag in cam2
    x_2_world = R_2 @ x_c2 + T_2              # convert to cam1 frame

    # Get tag pose from cam1 directly (already in world frame)
    x1, y1, z1 = cam1_pose
    x0, y0, z0 = cam0_pose
    x2, y2, z2 = cam2_pose

    csv_data.append([
        frame_indices[i],
        x_0_world[0, 0], x_0_world[1, 0], x_0_world[2, 0],
        x0, y0, z0,
        x1, y1, z1,
        x2, y2, z2,
        x_2_world[0, 0], x_2_world[1, 0], x_2_world[2, 0]
    ])

# ================= SAVE DATA TO CSV (ONCE) ================= #
csv_filename = tag_estimation_path
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Frame", "x_cam0_new", "y_cam0_new", "z_cam0_new", "x_cam0", "y_cam0", "z_cam0", "x_cam1", "y_cam1", "z_cam1", "x_cam2", "y_cam2", "z_cam2", "x_cam2_new", "y_cam2_new", "z_cam2_new"])  # Header
    writer.writerows(csv_data)  # Write all collected data at once

print(f"Saved stereo tag estimations to {csv_filename}")
