
import numpy as np
import cv2 as cv
im = cv.imread('img/standardleaves/1.jpg')

im = cv.blur(im, (8, 8))

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 150, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE,)
blank_image = np.zeros((imgray.shape[0],imgray.shape[0],3), np.uint8)
img = cv.drawContours(imgray, contours, -1, (0,255,0), 3)
img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()


print(len(contours))