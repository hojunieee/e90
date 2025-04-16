import cv2
import numpy as np
import glob
import os

saved_img_folder = "../E90/data/calibration/"
os.makedirs(saved_img_folder, exist_ok=True)
stereo = os.path.join(saved_img_folder, "stereo_imgs")
os.makedirs(stereo, exist_ok=True)
calibration_file = os.path.join("../E90/data/calibration/stereo_calibration.npz")

def enforce_consistent_order(corners, chessboard_size):
    if corners is None or len(corners) == 0:
        return None
    corners = corners.reshape(-1, 2)
    corners = corners.reshape(chessboard_size[1], chessboard_size[0], 2)
    corners = corners[np.argsort(corners[:, 0, 1])]
    for i in range(chessboard_size[1]):
        corners[i] = corners[i][np.argsort(corners[i, :, 0])]
    result = corners.reshape(-1, 1, 2)
    return result

# Take photos
for i in range(1):
    chessboard_size = (7, 5)
    square_size = 0.068
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # ==================== CAPTURE IMAGES ==================== #
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or more cameras.")
        exit()

    print("Press 's' to capture images, 'q' to quit.")
    # find the latest counter from this folder, and start from there
    cam0_images = sorted(glob.glob(os.path.join(stereo, "cam0_*.jpg")))
    cam1_images = sorted(glob.glob(os.path.join(stereo, "cam1_*.jpg")))
    img_counter = max(len(cam0_images), len(cam1_images))

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
            cv2.imwrite(os.path.join(stereo, f"cam0_{img_counter}.jpg"), frame0)
            cv2.imwrite(os.path.join(stereo, f"cam1_{img_counter}.jpg"), frame1)
            print(f"Saved set {img_counter + 1}")
            img_counter += 1
        elif key == ord('q'):
            print("Exiting...")
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

# ==================== LOAD IMAGES & DETECT CORNERS ==================== #


# Set chessboard pattern size

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) scaled by square_size
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []       # 3D points in world space
imgpoints_cam0 = []  # 2D points in cam0
imgpoints_cam1 = []  # 2D points in cam1

# Load image paths

cam0_images = sorted(glob.glob(os.path.join(stereo, "cam0_*.jpg")))
cam1_images = sorted(glob.glob(os.path.join(stereo, "cam1_*.jpg")))


assert len(cam0_images) == len(cam1_images), "Number of images must match"

# Detect chessboard corners
for img0_path, img1_path in zip(cam0_images, cam1_images):
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    ret0, corners0 = cv2.findChessboardCorners(gray0, chessboard_size, None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)

    expected_corners = chessboard_size[0] * chessboard_size[1]

    if (
        corners0 is None or corners1 is None or
        corners0.shape[0] != expected_corners or
        corners1.shape[0] != expected_corners
    ):
        print(f"Skipping due to mismatch in corners: {img0_path}, {img1_path}")
        continue

    corners0 = enforce_consistent_order(corners0, chessboard_size)
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

    if ret0 and ret1:
        objpoints.append(objp)
        imgpoints_cam0.append(corners0)
        imgpoints_cam1.append(corners1)
cv2.destroyAllWindows()

# Calibrate each camera individually
ret0, K_0, dist_0, _, _ = cv2.calibrateCamera(objpoints, imgpoints_cam0, gray0.shape[::-1], None, None)
ret1, K_1, dist_1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_cam1, gray1.shape[::-1], None, None)

# Stereo calibration with cam1 as the world frame
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retval, _, _, _, _, R_0, T_0, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_cam0, imgpoints_cam1,
    K_0, dist_0, K_1, dist_1,
    gray0.shape[::-1],
    criteria=criteria,
    flags=flags
)

# Output results
print("K_0:\n", K_0)
print("dist_0:\n", dist_0)
print("K_1:\n", K_1)
print("dist_1:\n", dist_1)
print("R_0 (cam0 to cam1/world):\n", R_0)
print("T_0 (cam0 to cam1/world):\n", T_0)

np.savez(calibration_file, 
             K_0=K_0, dist_0=dist_0,
             K_1=K_1, dist_1=dist_1,
             R_0=R_0, T_0=T_0)

print(f"Calibration results saved to {calibration_file}")