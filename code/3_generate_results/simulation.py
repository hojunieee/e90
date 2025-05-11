import cv2
import numpy as np
import apriltag
import os
import csv
import time

# === Setup ===
save_dir = "../E90/data/tag_capture_data/"
os.makedirs(save_dir, exist_ok=True)
img_dir = os.path.join(save_dir, "images")
os.makedirs(img_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "tag_positions.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'TagID', 'X', 'Y', 'Z', 'Timestamp'])

# === Calibration Load ===
calib = np.load("../E90/data/calibration/full_calibration.npz")
K = [calib['K_0'], calib['K_1'], calib['K_2']]
dist = [calib['dist_0'], calib['dist_1'], calib['dist_2']]
R = [calib['R_0'], calib['R_1'], calib['R_2']]
T = [calib['T_0'].reshape(3), calib['T_1'].reshape(3), calib['T_2'].reshape(3)]

# === AprilTag Setup ===
detector = apriltag.Detector()
o_points = np.array([[-0.158, 0.158, 0],
                     [ 0.158, 0.158, 0],
                     [ 0.158, -0.158, 0],
                     [-0.158, -0.158, 0]])

# === Previous World Coordinates per Camera-Tag ===
prev_positions = {}  # (cam_idx, tag_id) → np.array([x, y, z])
MAX_JUMP = 0.2  # meters

def detect_and_estimate(frame, K, dist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    results = []
    for det in detections:
        success, rvec, tvec = cv2.solvePnP(o_points, det.corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if success:
            results.append((det.tag_id, tvec.flatten()))
    return results

# === Camera Setup ===
caps = [cv2.VideoCapture(i) for i in [0, 4, 2]]
for cap in caps:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
try:
    while True:
        rets_frames = [cap.read() for cap in caps]
        if not all(ret for ret, _ in rets_frames):
            print("Camera error.")
            break

        frames = [frame for _, frame in rets_frames]
        all_detections = [
            detect_and_estimate(frames[0], K[0], dist[0]),
            detect_and_estimate(frames[1], K[1], dist[1]),
            detect_and_estimate(frames[2], K[2], dist[2])
        ]

        tag_positions = {}
        for cam_idx, detections in enumerate(all_detections):
            for tag_id, cam_coords in detections:
                world_coords = R[cam_idx].T @ (cam_coords - T[cam_idx])

                # Check for jump against last known position
                key = (cam_idx, tag_id)
                if key in prev_positions:
                    prev = prev_positions[key]
                    if np.linalg.norm(world_coords - prev) > MAX_JUMP:
                        continue  # Skip this outlier
                prev_positions[key] = world_coords

                if tag_id not in tag_positions:
                    tag_positions[tag_id] = []
                tag_positions[tag_id].append(world_coords)

        if frame_count % 5 == 0:
            timestamp = time.time()
            for tag_id, coords_list in tag_positions.items():
                if len(coords_list) >= 2:
                    avg_world = np.mean(coords_list, axis=0)
                    csv_writer.writerow([frame_count, tag_id, *avg_world, timestamp])

            for cam_idx, frame in enumerate(frames):
                resized = cv2.resize(frame, (1920, 1080))
                filename = os.path.join(img_dir, f"frame_{frame_count:05d}_cam{cam_idx}.png")
                cv2.imwrite(filename, resized)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    for cap in caps:
        cap.release()
    print(f"Data saved to: {save_dir}")
