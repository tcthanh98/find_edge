import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
im = cv.imread('./image/71.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 120, 255, 0)
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(im, contours, -1, (0, 255, 0), 3)

plt.imshow(im)
plt.show()
