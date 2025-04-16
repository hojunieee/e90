import cv2
import os
import numpy as np
import glob
import time

####################################################################
# Make sure to change saved_img_folder, output_file for each camera
saved_img_folder = "../E90/data/calibration/"
os.makedirs(saved_img_folder, exist_ok=True)
stereo = os.path.join(saved_img_folder, "stereo_imgs")
os.makedirs(stereo, exist_ok=True)

calibration_file = os.path.join("../E90/data/calibration/stereo_calibration.npz")
####################################################################

for i in range(1):

    chessboard_size = (7, 5)
    square_size = 0.068
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    def enforce_consistent_order(corners, chessboard_size):
        if corners is None or len(corners) == 0:
            return None
        corners = corners.reshape(-1, 2)
        corners = corners.reshape(chessboard_size[1], chessboard_size[0], 2)
        corners = corners[np.argsort(corners[:, 0, 1])]
        for i in range(chessboard_size[1]):
            corners[i] = corners[i][np.argsort(corners[i, :, 0])]
        return corners.reshape(-1, 1, 2)

    # ==================== CAPTURE IMAGES ==================== #
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or more cameras.")
        exit()

    print("Press 's' to capture images, 'q' to quit.")
    img_counter = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Error: Failed to capture images.")
            break

        combined_frame = np.hstack((frame0, frame1))
        cv2.imshow("Stereo Camera Feed", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(stereo, f"cam0_{timestamp}.jpg"), frame0)
            cv2.imwrite(os.path.join(stereo, f"cam1_{timestamp}.jpg"), frame1)
            print(f"Saved set {img_counter + 1}")
            img_counter += 1
        elif key == ord('q'):
            print("Exiting...")
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

# ==================== LOAD IMAGES & DETECT CORNERS ==================== #
objpoints = []
imgpoints_cam0 = []
imgpoints_cam1 = []

cam0_images = sorted(glob.glob(os.path.join(stereo, "cam0_*.jpg")))
cam1_images = sorted(glob.glob(os.path.join(stereo, "cam1_*.jpg")))

if len(cam0_images) != len(cam1_images):
    print("Mismatch in number of images between cam0 and cam1!")
    exit()

print(f"Found {len(cam0_images)} stereo image sets for calibration.")


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_shape = None

for img0_path, img1_path in zip(cam0_images, cam1_images):
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if image_shape is None:
        if gray0.shape != gray1.shape:
            raise Exception("Inconsistent image resolutions between cameras.")
        image_shape = gray0.shape[::-1]

    ret0, corners0 = cv2.findChessboardCorners(gray0, chessboard_size, None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)

    if ret0:
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
    if ret1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

    if ret0:
        corners0 = enforce_consistent_order(corners0, chessboard_size)
    if ret1:
        corners1 = enforce_consistent_order(corners1, chessboard_size)
    

    vis0 = cv2.drawChessboardCorners(img0.copy(), chessboard_size, corners0, ret0)
    vis1 = cv2.drawChessboardCorners(img1.copy(), chessboard_size, corners1, ret1)

    cv2.imshow("Chessboard Detection (Cam 0 | Cam 1)", np.hstack((vis0, vis1)))
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

    if ret0 and ret1:
        objpoints.append(objp)
        imgpoints_cam0.append(corners0)
        imgpoints_cam1.append(corners1)
    else:
        print(f"Skipping set due to detection failure: {img0_path}, {img1_path}")
cv2.destroyAllWindows()



# Print corners on images
print("Number of valid stereo pairs found: ", len(imgpoints_cam0))
# Print first five corners from each camera from the fisrt stereo pair
for i in range(5):
    print(f"Cam 0: frame {i}, point 0: {imgpoints_cam0[0][i]}")
    print(f"Cam 1: frame {i}, point 0: {imgpoints_cam1[0][i]}")
    print()




cam_matrix0 = np.zeros((3, 3))
cam_matrix1 = np.zeros((3, 3))
R_0 = np.zeros((3, 3))
T_0 = np.zeros((3, 1))

print(f"Number of valid stereo pairs used: {len(objpoints)}")

if len(imgpoints_cam0) > 0:
    ret, cam_matrix1, dist_coeffs1, cam_matrix0, dist_coeffs0, R_0, T_0, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_cam1, imgpoints_cam0,
        None, None, None, None,
        image_shape, flags=cv2.CALIB_RATIONAL_MODEL
    )

    fx = cam_matrix0[0, 0]
    fy = cam_matrix0[1, 1]

    print("Camera 0 Intrinsics:\n", cam_matrix0)
    print(f"Focal Lengths: fx = {fx}, fy = {fy}")
    print("Distortion Coefficients (Cam 0):\n", dist_coeffs0)
    print("Camera 1 Intrinsics:\n", cam_matrix1)
    print("Distortion Coefficients (Cam 1):\n", dist_coeffs1)
    print("Rotation Matrix of Cam 0:\n", R_0)
    print("Translation Vector of Cam 0):\n", T_0)

    np.savez(calibration_file, 
             cam_matrix0=cam_matrix0, dist_coeffs0=dist_coeffs0,
             cam_matrix1=cam_matrix1, dist_coeffs1=dist_coeffs1,
             R=R_0, T=T_0)

    print(f"Calibration results saved to {calibration_file}")
else:
    print("No valid stereo pairs found. Calibration aborted.")


# ============== TRIANGULATE POINTS ==============
# Projection matrices
P1 = cam_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Cam1 is world origin
P0 = cam_matrix0 @ np.hstack((R_0, T_0))                     # Cam0 relative to Cam1

triangulated_cam1_all = []
triangulated_cam0_all = []

for i, (pts1, pts0) in enumerate(zip(imgpoints_cam1, imgpoints_cam0)):
    pts1 = pts1.reshape(-1, 2).T  # shape: (2, N)
    pts0 = pts0.reshape(-1, 2).T  # shape: (2, N)

    points_4d = cv2.triangulatePoints(P1, P0, pts1, pts0)  # shape: (4, N)
    points_cam1 = (points_4d / points_4d[3])[:3].T         # shape: (N, 3)

    # Convert to Camera 0 frame: X_cam0 = R * X_cam1 + T
    points_cam0 = (R_0 @ points_cam1.T + T_0).T

    triangulated_cam1_all.append(points_cam1)
    triangulated_cam0_all.append(points_cam0)

    # Print comparison of corner 0 for first 5 frames
    if i < 5:
        print(f"\n[Frame {i}]")
        print(f"Corner 0 (Cam1 frame): {points_cam1[0]}")
        print(f"Corner 0 (Cam0 frame): {points_cam0[0]}")
