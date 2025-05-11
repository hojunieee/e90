import cv2
import numpy as np

img = np.zeros((300, 300), dtype=np.uint8)
cv2.putText(img, 'Hello', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()