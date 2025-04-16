import cv2
import os
import re

# Set folder to save images
save_folder = "../E90/data/calibration/captured_images_cam1"
os.makedirs(save_folder, exist_ok=True)

# Find the highest existing image index
def get_last_image_index(folder):
    image_files = [f for f in os.listdir(folder) if f.startswith("image_") and f.endswith(".jpg")]
    indices = [int(re.search(r"image_(\d+).jpg", f).group(1)) for f in image_files if re.search(r"image_(\d+).jpg", f)]
    return max(indices) + 1 if indices else 0

img_counter = get_last_image_index(save_folder)

# Open camera (0 for Cam 0, change if using other camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print(f"Starting from image_{img_counter}.jpg")
print("Press 's' to save a photo, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Display the live feed
    cv2.imshow("Camera Feed", frame)

    # Capture keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save image
        img_name = os.path.join(save_folder, f"image_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
