import cv2

camera_indices = [0, 1,2,3,4,5,6]
captures = []
window_names = []

try:
    for i, index in enumerate(camera_indices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            captures.append(cap)
            window_name = f"Camera {index}"
            window_names.append(window_name)
            cv2.namedWindow(window_name)
            print(f"Camera {index} opened successfully.")
        else:
            print(f"Error: Could not open camera {index}.")

    if not captures:
        print("No cameras were opened.")
    else:
        while True:
            frames = []
            all_frames_read = True
            for i, cap in enumerate(captures):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"Error: Could not read frame from {window_names[i]}.")
                    all_frames_read = False
                    break

            if not all_frames_read:
                break

            for i, frame in enumerate(frames):
                cv2.imshow(window_names[i], frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    for cap in captures:
        if cap.isOpened():
            cap.release()
    cv2.destroyAllWindows()