import numpy as np
import cv2

image = cv2.imread('timg.jpg', 0)
ker = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
cv2.imshow("before", image)

# aimage = np.zeros(image.shape)
aimage = cv2.filter2D(src=image, kernel=ker, ddepth=-1)

cv2.imshow("after", aimage)

cv2.waitKey(0)


