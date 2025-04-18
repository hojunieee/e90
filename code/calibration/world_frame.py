import cv2
import numpy as np
import glob
import os

# --- User input part --- #

# AprilTag length (side length, in meters)
tag_length = 0.16  # Example: 16 cm

# Center coordinates of four AprilTags in world frame
tag_centers_world = {
    0: np.array([0.0, 0.0, 0.0]),    # Tag ID 0 at (0,0,0)
    1: np.array([1.0, 0.0, 0.0]),    # Tag ID 1 at (1,0,0)
    2: np.array([1.0, 1.0, 0.0]),    # Tag ID 2 at (1,1,0)
    3: np.array([0.0, 1.0, 0.0]),    # Tag ID 3 at (0,1,0)
}

# Your camera intrinsics (already calibrated using chessboard)
# Load K_0, dist_0, K_1, dist_1, K_2, dist_2 from calibration file
calib = np.load("../E90/data/calibration/triple_calibration.npz")
K_0, dist_0 = calib["K_0"], calib["dist_0"]
K_1, dist_1 = calib["K_1"], calib["dist_1"]
K_2, dist_2 = calib["K_2"], calib["dist_2"]

# --- Helper functions --- #

def generate_apriltag_object_corners(center, tag_size):
    """Given a center, generate the four 3D corners in world frame."""
    half = tag_size / 2.0
    corners = np.array([
        [center[0] - half, center[1] - half, center[2]],  # Top-left
        [center[0] + half, center[1] - half, center[2]],  # Top-right
        [center[0] + half, center[1] + half, center[2]],  # Bottom-right
        [center[0] - half, center[1] + half, center[2]]   # Bottom-left
    ])
    return corners

def detect_apriltags(img):
    """Detect AprilTags using OpenCV."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, tag_family, parameters=detector_params)
    if ids is not None:
        return corners, ids
    else:
        return None, None

def estimate_camera_pose(img, K, dist):
    """Estimate camera pose relative to world frame using detected AprilTags."""
    corners, ids = detect_apriltags(img)

    if ids is None or len(ids) < 1:
        print("No AprilTags detected.")
        return None, None

    objpoints = []
    imgpoints = []

    ids = ids.flatten()
    for i, tag_id in enumerate(ids):
        if tag_id in tag_centers_world:
            world_corners = generate_apriltag_object_corners(tag_centers_world[tag_id], tag_length)
            img_corners = corners[i][0]  # Shape (4,2)

            objpoints.extend(world_corners)
            imgpoints.extend(img_corners)

    if len(objpoints) < 4:
        print("Not enough AprilTag corners for pose estimation.")
        return None, None

    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints = np.array(imgpoints, dtype=np.float32)

    # Solve PnP
    retval, rvec, tvec = cv2.solvePnP(
        objpoints, imgpoints, K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE if len(objpoints) == 4 else cv2.SOLVEPNP_ITERATIVE
    )

    # Invert to get World-to-Camera transformation
    R_cam_to_world, _ = cv2.Rodrigues(rvec)
    R_world_to_cam = R_cam_to_world.T
    t_world_to_cam = -R_world_to_cam @ tvec

    return R_world_to_cam, t_world_to_cam

# --- Main Process --- #

# Image paths to use for alignment
img_path_cam0 = "../E90/data/calibration/triple_imgs/cam0_0.jpg"
img_path_cam1 = "../E90/data/calibration/triple_imgs/cam1_0.jpg"
img_path_cam2 = "../E90/data/calibration/triple_imgs/cam2_0.jpg"

img0 = cv2.imread(img_path_cam0)
img1 = cv2.imread(img_path_cam1)
img2 = cv2.imread(img_path_cam2)

R_w0, T_w0 = estimate_camera_pose(img0, K_0, dist_0)
R_w1, T_w1 = estimate_camera_pose(img1, K_1, dist_1)
R_w2, T_w2 = estimate_camera_pose(img2, K_2, dist_2)

# Save the new extrinsics
np.savez("../E90/data/calibration/world_alignment.npz",
         R_w0=R_w0, T_w0=T_w0,
         R_w1=R_w1, T_w1=T_w1,
         R_w2=R_w2, T_w2=T_w2)

print("World alignment results saved to world_alignment.npz!")
