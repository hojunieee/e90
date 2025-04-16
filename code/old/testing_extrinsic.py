import cv2
import numpy as np
import os
import glob

# ==================== LOAD INTRINSIC AND EXTRINSIC PARAMETERS ==================== #
save_folder = "../E90/data/calibration/"
intrinsic_cam0 = os.path.join(save_folder, "camera_calibration_results_cam0.npz")
intrinsic_cam1 = os.path.join(save_folder, "camera_calibration_results_cam1.npz")
extrinsics_file = os.path.join(save_folder, "extrinsics.npz")

# Load intrinsic parameters
data0 = np.load(intrinsic_cam0)
K0 = data0["camera_matrix"]
dist0 = data0["dist_coeffs"]
K0_inv = np.linalg.inv(K0)

data1 = np.load(intrinsic_cam1)
K1 = data1["camera_matrix"]
dist1 = data1["dist_coeffs"]
K1_inv = np.linalg.inv(K1)

# Load extrinsic parameters
data = np.load(extrinsics_file)
R = data["R"]  # Rotation matrix Cam 1 → Cam 0
T = data["T"]  # Translation vector Cam 1 → Cam 0

print("Loaded camera matrices and extrinsics.")
print("K0:", K0)
print("K1:", K1)
print("R:", R)
print("T:", T)

# ==================== LOAD EXISTING IMAGE PAIRS ==================== #
stereo = os.path.join(save_folder, "stereo_imgs")

# Load captured images
cam0_images = sorted(glob.glob(os.path.join(stereo, "cam0_*.jpg")))
cam1_images = sorted(glob.glob(os.path.join(stereo, "cam1_*.jpg")))

if len(cam0_images) == 0 or len(cam1_images) == 0:
    raise Exception("No images found!")

if len(cam0_images) != len(cam1_images):
    raise Exception("Mismatch in number of images between cam0 and cam1!")

# Select a test image pair (e.g., the first pair)
img0_path = cam0_images[0]
img1_path = cam1_images[0]

img0 = cv2.imread(img0_path)
img1 = cv2.imread(img1_path)

# Convert images to grayscale
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# ==================== DETECT CHESSBOARD CORNERS ==================== #
chessboard_size = (7, 7)  # Inner corners
ret0, corners0 = cv2.findChessboardCorners(gray0, chessboard_size, None)
ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)

if not ret0 or not ret1:
    raise Exception("Chessboard not found in both images!")

# ==================== TRANSFORM img1_corners TO img1_corners_new ==================== #
img1_corners_new = []

point_cam1 = corners1[0][0] # this should be (u, v) in pixel coordinates
print("point_cam1:", point_cam1)
point_cam1_hom = np.array([point_cam1[0], point_cam1[1], 1.0])
print("point_cam1_hom:", point_cam1_hom)
point_cam1_3D = (K1_inv @ point_cam1_hom).reshape(3, 1)  # Ensure it's a (3,1) column vector
print("point_cam1_3D:", point_cam1_3D)
point_cam0_3D = R @ point_cam1_3D + T  # Transform to cam0
print("point_cam0_3D:", point_cam0_3D)
point_cam1_new_hom = K0 @ (point_cam0_3D / point_cam0_3D[2])  # Projection onto cam0's image plane
print("point_cam1_new_hom:", point_cam1_new_hom)
point_cam1_new_on_img0 = point_cam1_new_hom[:2].flatten()
print("point_cam1_new_on_img0:", point_cam1_new_on_img0)

print("=====================================")
print("Moved from ", point_cam1, " to ", point_cam1_new_on_img0)


# # Convert to NumPy array for easier processing
# img1_corners_new = np.array(img1_corners_new, dtype=np.float32)

# # ==================== VISUALIZE RESULTS ==================== #
# img0_corners = img0.copy()
# img1_corners = img1.copy()
# img0_with_img1_corners_new = img0.copy()  # img0 with transformed cam1 corners

# # Draw detected chessboard corners in images
# cv2.drawChessboardCorners(img0_corners, chessboard_size, corners0, ret0)
# cv2.drawChessboardCorners(img1_corners, chessboard_size, corners1, ret1)


# print("Transformed corner coordinates on img0:")
# for idx, pt in enumerate(img1_corners_new):
#     x, y = int(pt.flatten()[0]), int(pt.flatten()[1])
#     print(f"Point {idx}: x={x}, y={y}")

# print("K0:", K0)
# print("K1:", K1)

# print("R:", R)
# print("T:", T)

# # Draw transformed points from Camera 1 on Camera 0's image
# for pt in img1_corners_new:
#     x, y = int(pt.flatten()[0]), int(pt.flatten()[1])  # Flatten and extract values
#     cv2.circle(img0_with_img1_corners_new, (x, y), 5, (0, 255, 0), -1)  # Green dots

# # Ensure all images have the same height before stacking
# h = min(img0_corners.shape[0], img1_corners.shape[0], img0_with_img1_corners_new.shape[0])

# img0_corners_resized = cv2.resize(img0_corners, (int(img0_corners.shape[1] * h / img0_corners.shape[0]), h))
# img1_corners_resized = cv2.resize(img1_corners, (int(img1_corners.shape[1] * h / img1_corners.shape[0]), h))
# img0_with_img1_corners_new_resized = cv2.resize(img0_with_img1_corners_new, 
#                                                  (int(img0_with_img1_corners_new.shape[1] * h / img0_with_img1_corners_new.shape[0]), h))

# # Stack images horizontally
# comparison_image = np.hstack((img0_corners_resized, img1_corners_resized, img0_with_img1_corners_new_resized))

# # Show the stacked image
# cv2.imshow("Stacked Images (Left: img0, Middle: img1, Right: img1 Corners on img0)", comparison_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
