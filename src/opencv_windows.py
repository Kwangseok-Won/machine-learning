import cv2
import numpy as np

# Check version:
print(cv2.__version__)
# Create a test image and show it:
img = np.zeros((300, 400, 3), dtype=np.uint8)
_ = cv2.putText(
    img, "OpenCV works!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
)
cv2.imshow("Test", img)
_ = cv2.waitKey(0)
cv2.destroyAllWindows()
