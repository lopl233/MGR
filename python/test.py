
import numpy as np
import cv2 as cv
im = cv.imread('img/standardleaves/16.jpg')

im = cv.blur(im, (20, 20))

im = cv.resize(im, (0,0), fx=0.5, fy=0.5)


imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 200, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE,)
blank_image = np.zeros((imgray.shape[0],imgray.shape[1],3), np.uint8)


contours_len = [len(x) for x in contours]
contours = [x for x in contours if len(x) == max(contours_len) ]

img = cv.drawContours(blank_image, contours, -1, (255,255,255), 3)

img = cv.blur(img, (20,20))

print(imgray[0][0])

for x in range(imgray.shape[0]):
    for y in range(imgray.shape[1]):
        if not any(img[x][y]) or imgray[x][y] > 200:
            imgray[x][y] = 255


im_color = cv.applyColorMap(imgray ,cv.COLORMAP_JET)

print(im_color[0][0])

for x in range(imgray.shape[0]):
    for y in range(imgray.shape[1]):
        if im_color[x][y][0] == 0 and im_color[x][y][1] == 0:
            im_color[x][y] = 255

cv.imshow('image',im_color)
cv.waitKey(0)
cv.destroyAllWindows()


