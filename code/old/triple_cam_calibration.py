import cv2
import numpy as np
import os
import time
import glob

# ==================== SETUP ==================== #
save_folder = "../E90/data/calibration/"
os.makedirs(save_folder, exist_ok=True)
triple = os.path.join(save_folder, "triple_imgs")
os.makedirs(triple, exist_ok=True)

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
cap2 = cv2.VideoCapture(2)

if not cap0.isOpened() or not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or more cameras.")
    exit()

print("Press 's' to capture images, 'q' to quit.")
img_counter = 0

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret1 or not ret2:
        print("Error: Failed to capture images.")
        break

    combined_frame = np.hstack((frame0, frame1, frame2))
    cv2.imshow("Triple Camera Feed", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(stereo, f"cam0_{timestamp}.jpg"), frame0)
        cv2.imwrite(os.path.join(stereo, f"cam1_{timestamp}.jpg"), frame1)
        cv2.imwrite(os.path.join(stereo, f"cam2_{timestamp}.jpg"), frame2)
        print(f"Saved set {img_counter + 1}")
        img_counter += 1
    elif key == ord('q'):
        print("Exiting...")
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# ==================== LOAD IMAGES & CALIBRATE ==================== #
objpoints = []
imgpoints_cam0 = []
imgpoints_cam1 = []
imgpoints_cam2 = []

cam0_images = sorted(glob.glob(os.path.join(triple, "cam0_*.jpg")))
cam1_images = sorted(glob.glob(os.path.join(triple, "cam1_*.jpg")))
cam2_images = sorted(glob.glob(os.path.join(triple, "cam2_*.jpg")))

if len(cam0_images) != len(cam1_images) or len(cam0_images) != len(cam2_images):
    raise Exception("Mismatch in number of images from all cameras!")

print(f"Found {len(cam0_images)} image sets for calibration.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_shape = None
for img0_path, img1_path, img2_path in zip(cam0_images, cam1_images, cam2_images):
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if image_shape is None:
        if gray0.shape != gray1.shape or gray0.shape != gray2.shape:
            raise Exception("Inconsistent image resolutions among cameras.")
        image_shape = gray0.shape[::-1]

    ret0, corners0 = cv2.findChessboardCorners(gray0, chessboard_size, None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if ret0:
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
    if ret1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    if ret2:
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    if ret0:
        corners0 = enforce_consistent_order(corners0, chessboard_size)
    if ret1:
        corners1 = enforce_consistent_order(corners1, chessboard_size)
    if ret2:
        corners2 = enforce_consistent_order(corners2, chessboard_size)

    vis0 = cv2.drawChessboardCorners(img0.copy(), chessboard_size, corners0, ret0)
    vis1 = cv2.drawChessboardCorners(img1.copy(), chessboard_size, corners1, ret1)
    vis2 = cv2.drawChessboardCorners(img2.copy(), chessboard_size, corners2, ret2)

    cv2.imshow("Chessboard Detection (Cam 0 | Cam 1 | Cam 2)", np.hstack((vis0, vis1, vis2)))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    if ret0 and ret1 and ret2:
        objpoints.append(objp)
        imgpoints_cam0.append(corners0)
        imgpoints_cam1.append(corners1)
        imgpoints_cam2.append(corners2)
    else:
        print(f"Skipping set due to detection failure: {img0_path}, {img1_path}, {img2_path}")

cv2.destroyAllWindows()

# ==================== STEREO CALIBRATION ==================== #
ret1, cam_matrix1, dist_coeffs1, cam_matrix0, dist_coeffs0, R_1_0, T_1_0, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints_cam1, imgpoints_cam0,
    None, None, None, None,
    image_shape, flags=cv2.CALIB_RATIONAL_MODEL
)

# Initial guesses for Camera 2
cam_matrix2_init = np.eye(3, dtype=np.float64)
dist_coeffs2_init = np.zeros((14, 1), dtype=np.float64)

flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL

# ret2, _, _, cam_matrix2, dist_coeffs2, R_1_2, T_1_2, _, _ = cv2.stereoCalibrate(
#     objpoints, imgpoints_cam1, imgpoints_cam2,
#     cam_matrix1, dist_coeffs1,           # Use Cam1's calibrated intrinsics
#     cam_matrix2_init, dist_coeffs2_init, # Guesses for Cam2
#     image_shape,
#     flags=flags
# )

cam_matrix2 = cam_matrix1
dist_coeffs2 = dist_coeffs1
R_1_2 = np.eye(3)
T_1_2 = np.zeros((3, 1))

# ==================== SAVE AND PRINT RESULTS ==================== #
calibration_file = os.path.join(save_folder, "triple_calibration.npz")
np.savez(calibration_file,
         cam_matrix0=cam_matrix0, dist_coeffs0=dist_coeffs0,
         cam_matrix1=cam_matrix1, dist_coeffs1=dist_coeffs1,
         cam_matrix2=cam_matrix2, dist_coeffs2=dist_coeffs2,
         R_1_0=R_1_0, T_1_0=T_1_0,
         R_1_2=R_1_2, T_1_2=T_1_2)

print(f"Calibration parameters saved to {calibration_file}")
print("Camera 0 Intrinsics:\n", cam_matrix0)
print("Camera 0 Distortion Coefficients:\n", dist_coeffs0)
print("Camera 1 Intrinsics:\n", cam_matrix1)
print("Camera 1 Distortion Coefficients:\n", dist_coeffs1)
# print("Camera 2 Intrinsics:\n", cam_matrix2)
# print("Camera 2 Distortion Coefficients:\n", dist_coeffs2)
print("Rotation Matrix from Camera 1 to Camera 0:\n", R_1_0)
print("Translation Vector from Camera 1 to Camera 0:\n", T_1_0)
# print("Rotation Matrix from Camera 1 to Camera 2:\n", R_1_2)
# print("Translation Vector from Camera 1 to Camera 2:\n", T_1_2)
