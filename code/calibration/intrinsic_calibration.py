import cv2
import numpy as np
import os
import glob

# === Configuration ===

camera_id = 0  # Choose the camera (4->Top, 0->Front, 2->Back)
cap = cv2.VideoCapture(camera_id)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
chessboard_size = (4, 3)  # Chessboard size (columns, rows)
square_size = 0.108  # Size of each square on the chessboard in meters

saved_img_folder = "../E90/data/calibration/"
os.makedirs(saved_img_folder, exist_ok=True)
mono = os.path.join(saved_img_folder, "mono_imgs")
os.makedirs(mono, exist_ok=True)

img_prefix = f"cam{camera_id}"
img_format = f"{img_prefix}_{{:02d}}.jpg"
img_save_folder = os.path.join(mono, f"{img_prefix}_images/")
os.makedirs(img_save_folder, exist_ok=True)

calibration_file = os.path.join(saved_img_folder, f"mono_{img_prefix}.npz")


# === Step 1: Capture images ===
if not cap.isOpened():
    raise IOError(f"Cannot open camera {camera_id}")

print("Press 's' to save an image, or 'q' to quit.")

img_counter = len(glob.glob(os.path.join(img_save_folder, "cam0_*.jpg")))
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        save_path = os.path.join(img_save_folder, img_format.format(img_counter))
        cv2.imwrite(save_path, frame)
        print(f"Saved {save_path}")
        img_counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Step 2: Camera Calibration ===
image_paths = sorted(glob.glob(os.path.join(img_save_folder, f"{img_prefix}_*.jpg")))

# Prepare 3D object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners_subpix = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners_subpix)
        vis = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners_subpix, ret)
        cv2.imshow("Detected Corners", vis)
        print("Press any key to continue or 'q' to quit preview.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    else:
        print(f"Chessboard not found in {path}")

cv2.destroyAllWindows()

# Perform calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Compute reprojection error
def compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)
        total_error += error**2
        total_points += len(objpoints[i])
    return np.sqrt(total_error / total_points)

error = compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)

# Output calibration results
print("Calibration complete")
print("K (intrinsics):\n", K)
print("Distortion coefficients:\n", dist)
print(f"Reprojection error: {error:.4f} pixels")

# Save the results
np.savez(calibration_file, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)
print(f"Calibration results saved for camera {camera_id}")
