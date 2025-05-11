import cv2
import numpy as np
import apriltag
import os
import csv
import time

# ==================== Setup Paths ==================== #
save_folder = "../E90/data/"
calib_folder = os.path.join(save_folder, "calibration")
output_csv = os.path.join(save_folder, "triple_tag_estimations.csv")
os.makedirs(save_folder, exist_ok=True)

# ==================== Load Calibration ==================== #
full_calib = np.load(os.path.join(calib_folder, "full_calibration.npz"))
K0, dist0, R0, T0 = full_calib["K_0"], full_calib["dist_0"], full_calib["R_0"], full_calib["T_0"].reshape(3)
K1, dist1, R1, T1 = full_calib["K_1"], full_calib["dist_1"], full_calib["R_1"], full_calib["T_1"].reshape(3)
K2, dist2, R2, T2 = full_calib["K_2"], full_calib["dist_2"], full_calib["R_2"], full_calib["T_2"].reshape(3)

camera_params = [
    (0, K0, dist0, R0, T0),
    (4, K1, dist1, R1, T1),
    (2, K2, dist2, R2, T2),
]

# ==================== AprilTag Setup ==================== #
tag_size = 0.158  # meters
tag_object_pts = np.array([
    [-tag_size/2, -tag_size/2, 0],
    [ tag_size/2, -tag_size/2, 0],
    [-tag_size/2,  tag_size/2, 0],
    [ tag_size/2,  tag_size/2, 0],
], dtype=np.float32)

detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

# ==================== Camera Streams ==================== #
caps = {}
for cam_id, _, _, _, _ in camera_params:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    caps[cam_id] = cap

# ==================== Helpers ==================== #
def reorder_corners(corners, cam_id):
    pts = corners.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    if cam_id in (1, 4):
        order = [np.argmin(s), np.argmin(diff), np.argmax(diff), np.argmax(s)]
    elif cam_id == 2:
        order = [np.argmin(diff), np.argmax(s), np.argmin(s), np.argmax(diff)]
    else:  # cam_id == 0
        order = [np.argmax(diff), np.argmin(s), np.argmax(s), np.argmin(diff)]
    return pts[order].astype(np.float32)

# ==================== Tracking Loop ==================== #
rows = []
frame_idx = 0

try:
    while True:
        frame_entry = [frame_idx]
        seen = set()

        for cam_id, K, dist, R, T in camera_params:
            ret, frame = caps[cam_id].read()
            if not ret:
                frame_entry += [np.nan]*6
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray)
            if not dets:
                frame_entry += [np.nan]*6
                continue

            det = dets[0]
            corners = reorder_corners(det.corners, cam_id)
            success, rvec, tvec = cv2.solvePnP(
                tag_object_pts, corners, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                frame_entry += [np.nan]*6
                continue

            cam_coords = tvec.flatten()
            # === correct world‐frame transform ===
            world_coords = R.T @ (cam_coords - T)

            # pack: [x_new, y_new, z_new, x_cam, y_cam, z_cam]
            xw, yw, zw = world_coords
            xc, yc, zc = cam_coords
            if cam_id == 0:
                frame_entry += [xw, yw, zw, xc, yc, zc]
            elif cam_id == 4:
                frame_entry += [xw, yw, zw, xc, yc, zc]
            else:  # cam_id == 2
                frame_entry += [xc, yc, zc, xw, yw, zw]

            # visualize
            vis = frame.copy()
            for i, pt in enumerate(corners):
                p = tuple(pt.astype(int))
                cv2.circle(vis, p, 5, (0,255,0), -1)
                cv2.putText(vis, str(i+1), (p[0]+5,p[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            c = tuple(det.center.astype(int))
            cv2.circle(vis, c, 5, (0,0,255), -1)
            cv2.putText(vis, f"Cam{cam_id}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow(f"Camera {cam_id}", vis)

            seen.add(cam_id)

        # only save frames with no NaNs
        if seen and not any(np.isnan(frame_entry[1:])):
            rows.append(frame_entry)
            frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Frame",
            "x_cam0_new","y_cam0_new","z_cam0_new","x_cam0","y_cam0","z_cam0",
            "x_cam1_new","y_cam1_new","z_cam1_new","x_cam1","y_cam1","z_cam1",
            "x_cam2","y_cam2","z_cam2","x_cam2_new","y_cam2_new","z_cam2_new"
        ])
        w.writerows(rows)

    print(f"Saved {len(rows)} frames to {output_csv}")
