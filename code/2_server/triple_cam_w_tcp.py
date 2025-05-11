import cv2
import apriltag
import numpy as np
import os
import time
import socket
import json
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# ==================== Setup ==================== #
save_folder = "../E90/data/"
camera_param = os.path.join(save_folder, "calibration/triple_calibration.npz")
frame_save_dir = os.path.join(save_folder, "calibration/triple_tag_frames")
os.makedirs(frame_save_dir, exist_ok=True)

if os.path.exists(camera_param):
    data = np.load(camera_param)
    K0, K1, K2 = data['K_0'], data['K_1'], data['K_2']
    dist_coeffs0, dist_coeffs1, dist_coeffs2 = data['dist_0'], data['dist_1'], data['dist_2']
    R_0, T_0 = data['R_0'], data['T_0']
    R_2, T_2 = data['R_2'], data['T_2']
else:
    raise FileNotFoundError(f"Calibration file not found at {camera_param}")

tagsize = 0.09  # meters
o_points = np.array([[-tagsize/2, tagsize/2, 0], 
                     [tagsize/2, tagsize/2, 0], 
                     [tagsize/2, -tagsize/2, 0], 
                     [-tagsize/2, -tagsize/2, 0]])

detector = apriltag.Detector()
cap0, cap1, cap2 = cv2.VideoCapture(0), cv2.VideoCapture(1), cv2.VideoCapture(2)
if not all([cap0.isOpened(), cap1.isOpened(), cap2.isOpened()]):
    print("Error: Could not open all cameras.")
    exit()

HOST, PORT = "0.0.0.0", 5006
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"TCP Server started on {HOST}:{PORT}")

server.settimeout(5.0)
try:
    client, addr = server.accept()
    print(f"Connected to {addr}")
except socket.timeout:
    print("No client connected. Proceeding without TCP.")
    client = None

# ==================== Tag Detection ==================== #
def detect_tags(frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    with suppress_stdout_stderr():
        detections = detector.detect(gray)
    results = []
    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners
        success, rvec, tvec = cv2.solvePnP(
            o_points, corners, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if success:
            position_3d = tvec.flatten()
            center_2d = detection.center
            results.append((tag_id, position_3d, center_2d))
    return results

# ==================== Main Loop ==================== #
fps_list = []
frame_idx = 0
prev_positions = {}  # tag_id 
try:
    while True:
        start_time = time.time()
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not all([ret0, ret1, ret2]):
            print("Frame capture failed.")
            break

        det0 = detect_tags(frame0, K0, dist_coeffs0)
        det1 = detect_tags(frame1, K1, dist_coeffs1)
        det2 = detect_tags(frame2, K2, dist_coeffs2)

        det0_dict = {tag_id: pos for tag_id, pos, _ in det0}
        det1_dict = {tag_id: pos for tag_id, pos, _ in det1}
        det2_dict = {tag_id: pos for tag_id, pos, _ in det2}

        common_ids = set(det0_dict) & set(det1_dict) & set(det2_dict)

        if not client:
            try:
                client, addr = server.accept()
            except socket.timeout:
                pass

        for tag_id in common_ids:
            x_0 = R_0 @ det0_dict[tag_id].reshape(3, 1) + T_0
            x_1 = det1_dict[tag_id].reshape(3, 1)
            x_2 = R_2 @ det2_dict[tag_id].reshape(3, 1) + T_2
        
            # Store positions
            positions = [x_0.flatten(), x_1.flatten(), x_2.flatten()]
        
            # Outlier rejection: reject any detection that jumps >0.1m from previous detection
            valid_positions = []
            for pos in positions:
                if tag_id in prev_positions:
                    prev_pos = prev_positions[tag_id]
                    if np.linalg.norm(pos - prev_pos) < 0.1:
                        valid_positions.append(pos)
                else:
                    valid_positions.append(pos)  # no history yet
        
            if len(valid_positions) == 0:
                continue  # skip if all rejected
        
            avg_pos = np.mean(valid_positions, axis=0)
            prev_positions[tag_id] = avg_pos  # update history
        
            time_stamp = time.time()
            tag_info = {
                "tagID": int(tag_id),
                "position": avg_pos.flatten().tolist(),
                "timestamp": time_stamp,
            }
            message = json.dumps(tag_info) + "\n"
        
            if client:
                try:
                    client.sendall(message.encode('utf-8'))
                except (BrokenPipeError, ConnectionResetError):
                    client.close()
                    client = None

        if client and frame_idx % 60 == 0 and not common_ids:
            dummy_info = {
                "tagID": -1,
                "position": [0, 0, 0],
                "timestamp": time.time()
            }
            try:
                client.sendall((json.dumps(dummy_info) + "\n").encode('utf-8'))
            except:
                client = None

        frame0 = cv2.resize(frame0, (320, 240))
        frame1 = cv2.resize(frame1, (320, 240))
        frame2 = cv2.resize(frame2, (320, 240))
        combined = np.hstack((frame0, frame1, frame2))
        cv2.imshow("Triple Cam View", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps = 1 / (time.time() - start_time)
        fps_list.append(fps)
        frame_idx += 1

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap0.release()
    cap1.release()
    cap2.release()
    if client:
        client.close()
    server.close()
    cv2.destroyAllWindows()

    print("########################################")
    print(f"Average FPS: {np.mean(fps_list):.2f}")
    print(f"Total frames processed: {frame_idx}")
    print("########################################")
