import cv2
import numpy as np
import apriltag
import os

# ----------------------------- #
# Load camera intrinsic matrices
# ----------------------------- #
saved_img_folder = "../E90/data/calibration/"
os.makedirs(saved_img_folder, exist_ok=True)

mono_cam0 = np.load(os.path.join(saved_img_folder, "mono_cam0.npz"))
mono_cam1 = np.load(os.path.join(saved_img_folder, "mono_cam4.npz"))
mono_cam2 = np.load(os.path.join(saved_img_folder, "mono_cam2.npz"))

K0 = mono_cam0["K"]
dist0 = mono_cam0["dist"]

K1 = mono_cam1["K"]
dist1 = mono_cam1["dist"]

K2 = mono_cam2["K"]
dist2 = mono_cam2["dist"]
##############################

scale_x = 1920 / 1280
scale_y = 1080 / 720
def rescale_K(K, scale_x, scale_y):
    K = K.copy()
    K[0,0] *= scale_x
    K[0,2] *= scale_x
    K[1,1] *= scale_y
    K[1,2] *= scale_y
    return K

# K0 = rescale_K(K0, scale_x, scale_y)
# K1 = rescale_K(K1, scale_x, scale_y)
# K2 = rescale_K(K2, scale_x, scale_y)
##############################

# Camera list
camera_params = [
    (K0, dist0, 0),
    (K1, dist1, 4),
    (K2, dist2, 2)
]

# ----------------------------- #
# Define AprilTag properties
# ----------------------------- #
half_size = 0.155  # half of 0.31
tag_centers = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([0.76, 0.5, 0.0]),
    3: np.array([0.0, 0.5, 0.0]),
    4: np.array([0.76, 0.0, 0.0]),
}

def tag_corners(center):
    cx, cy, cz = center
    return [
        [cx - half_size, cy - half_size, cz],  # TL
        [cx + half_size, cy - half_size, cz],  # TR
        [cx - half_size, cy + half_size, cz],  # BL
        [cx + half_size, cy + half_size, cz],  # BR   
    ]

# Append world_points the way we read (first tags, then within tags too)
world_points = []
for tag_id in [1, 2, 3, 4]:  # order you specified
    world_points.extend(tag_corners(tag_centers[tag_id]))
print(world_points)


# ----------------------------- #
# Setup AprilTag Detector
# ----------------------------- #
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# read detection from a single tag, then append in a way that is consistent with world frame
def reorder_corners(corners, cam_id):
    # corners: (4,2) numpy array
    pts = corners.reshape((4, 2))
    rect = []

    s = pts.sum(axis=1)       # x + y
    diff = np.diff(pts, axis=1)  # y - x

    if cam_id in (1, 4):
        rect.append(pts[np.argmin(s)])     # top-left
        rect.append(pts[np.argmin(diff)])  # top-right
        rect.append(pts[np.argmax(diff)])  # bottom-left 
        rect.append(pts[np.argmax(s)])     # bottom-right

    elif cam_id == 2:
        rect.append(pts[np.argmin(diff)])  # top-right
        rect.append(pts[np.argmax(s)])     # bottom-right
        rect.append(pts[np.argmin(s)])     # top-left
        rect.append(pts[np.argmax(diff)])  # bottom-left
    
    elif cam_id == 0:
        rect.append(pts[np.argmax(diff)])  # bottom-left
        rect.append(pts[np.argmin(s)])     # top-left
        rect.append(pts[np.argmax(s)])     # bottom-right
        rect.append(pts[np.argmin(diff)])  # top-right

    return rect  # output as list

def get_correspondences(img_gray, cam_id):
    detections = detector.detect(img_gray)

    obj_pts = []
    img_pts = []

    for det in detections:
        print(det.tag_id)
        tag_id = det.tag_id
        if tag_id in tag_centers:
            # Get corresponding 3D world corners
            world_corners = tag_corners(tag_centers[tag_id])   # 4 world points
            
            # Reorder 2D image corners
            img_corners = reorder_corners(det.corners, cam_id) # 4 image points

            if len(img_corners) != 4:
                raise AssertionError("Not all corners were detected for tag", tag_id)

            # Append correctly
            for wc, ic in zip(world_corners, img_corners):
                obj_pts.append(wc)
                img_pts.append(ic)

    if len(obj_pts) == 0 or len(img_pts) == 0:
        print(f"No valid detections for camera {cam_id}")
        return None, None

    if len(obj_pts) != len(img_pts):
        raise AssertionError(f"Mismatch: {len(obj_pts)} object points vs {len(img_pts)} image points!")

    return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)


# ----------------------------- #
# Main calibration loop
# ----------------------------- #
camera_extrinsics = {}

for K, dist, cam_id in camera_params:
    # Open the corresponding camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080 )
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"Camera {cam_id} opened. Press 's' to capture frame for calibration.")

    # Display capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {cam_id}")
            continue

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(img_gray)

        display = frame.copy()

        # Draw detection results
        corner_idx = 0  # Global counter from 0 to 15
        for det in detections:
            tag_id = det.tag_id
            if tag_id in tag_centers:
                # Reorder corners properly before displaying
                img_corners = reorder_corners(det.corners, cam_id)

                for corner in img_corners:
                    corner = corner.astype(int)
                    cv2.circle(display, tuple(corner), 5, (255, 0, 0), -1)
                    cv2.putText(display, str(corner_idx), tuple(corner + np.array([5, -5])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    corner_idx += 1

            # Draw tag center and ID for visualization
            center = tuple(det.center.astype(int))
            cv2.putText(display, f"ID:{tag_id}", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw pixel frame origin and x,y axes
        origin = (50, 50)  # origin of pixel coordinate frame
        x_axis = (origin[0] + 100, origin[1])  # x direction: right
        y_axis = (origin[0], origin[1] + 100)  # y direction: down

        # Draw x-axis
        cv2.arrowedLine(display, origin, x_axis, (255, 0, 0), 2, tipLength=0.2)
        cv2.putText(display, 'x', (x_axis[0] + 10, x_axis[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw y-axis
        cv2.arrowedLine(display, origin, y_axis, (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(display, 'y', (y_axis[0], y_axis[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mark origin
        cv2.circle(display, origin, 5, (0, 0, 255), -1)
        cv2.putText(display, 'origin', (origin[0] - 40, origin[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(display, f"Camera {cam_id}: Press 's' to capture", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow(f'Camera {cam_id}', display)

        key = cv2.waitKey(1)
        if key == ord('s'):
            print(f"Captured image for camera {cam_id}")
            captured_frame = frame.copy()  # Save the captured frame
            captured_gray = img_gray.copy()
            break
        elif key == ord('q'):  # q to exit
            print("User exited.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()

    # Find correspondences on captured frame
    obj_pts, img_pts = get_correspondences(captured_gray, cam_id)
    if obj_pts is None:
        print(f"No valid detections for camera {cam_id}. Skipping.")
        continue

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        print(f"SolvePnP failed for Camera {cam_id}")
        continue

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Store
    camera_extrinsics[f'cam{cam_id}'] = {
        'R': R,
        'T': tvec
    }

    print(f"Camera {cam_id} calibration done.")
    print(f"R:\n{R}")
    print(f"T:\n{tvec.flatten()}")

# ----------------------------- #
# Save extrinsic and intrinsic parameters
# ----------------------------- #
save_dict = {
    'R_0': camera_extrinsics['cam0']['R'],
    'T_0': camera_extrinsics['cam0']['T'],
    'R_1': camera_extrinsics['cam4']['R'],
    'T_1': camera_extrinsics['cam4']['T'],
    'R_2': camera_extrinsics['cam2']['R'],
    'T_2': camera_extrinsics['cam2']['T'],
    'K_0': K0,
    'dist_0': dist0,
    'K_1': K1,
    'dist_1': dist1,
    'K_2': K2,
    'dist_2': dist2
}

np.savez(os.path.join(saved_img_folder, 'full_calibration_new.npz'), **save_dict)
print(f"Extrinsic and intrinsic parameters saved to {saved_img_folder}/full_calibration.npz")

# ----------------------------- #
# Visualize Reprojection
# ----------------------------- #


def project_world_points(world_pts, K, dist, R, T):
    """
    Project 3D world points into 2D image plane using given camera parameters.
    """
    world_pts = np.array(world_pts, dtype=np.float32).reshape(-1, 3)
    
    # Project points
    img_pts, _ = cv2.projectPoints(world_pts, cv2.Rodrigues(R)[0], T, K, dist)

    img_pts = img_pts.reshape(-1, 2)
    return img_pts

for cam_id, (K, dist) in zip([0, 4, 2], [(K0, dist0), (K1, dist1), (K2, dist2)]):
    # Load the extrinsics
    R = camera_extrinsics[f'cam{cam_id}']['R']
    T = camera_extrinsics[f'cam{cam_id}']['T']

    # Open camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id} for reprojection check.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to capture frame for reprojection visualization on camera {cam_id}")
        continue

    # Project world points
    img_pts = project_world_points(world_points, K, dist, R, T)
    # Also collect AprilTag detections to compare
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_obj_pts, detected_img_pts = get_correspondences(img_gray, cam_id)

    if detected_img_pts is not None:
        for i, pt in enumerate(detected_img_pts):
            pt = tuple(int(x) for x in pt)
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)  # Blue dot for detected
            cv2.putText(frame, str(i), (pt[0] - 15, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw the reprojected points
    for i, pt in enumerate(img_pts):
        pt = tuple(int(x) for x in pt)
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)  # Red dot for reprojected
        cv2.putText(frame, str(i), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.putText(frame, f"Camera {cam_id}: Reprojected world points", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(f"Reprojection - Camera {cam_id}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()