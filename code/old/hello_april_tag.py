import cv2
import apriltag
import matplotlib.pyplot as plt

# Load the image and initialize
# image_path = "test/img/tag41_12_00000.png"
image_path = "test/data/test_img3.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Failed to load image at {image_path}. Check the path.")

# plot image
plt.imshow(image)

detector = apriltag.Detector()

detections = detector.detect(image)

# Print detected tags
if detections:
    for detection in detections:
        print(f"Detected tag ID: {detection.tag_id} at {detection.center}")
        print(f"Tag corners: {detection.corners}")
else:
    print("No AprilTags detected.")
