import numpy as np
import imutils
import cv2 as cv

def crop_image(img,tol=0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    return crop

def rotateUntilBest(image):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (3, 3), 0)
	ret, edged = cv.threshold(gray, 240, 255, cv.THRESH_BINARY)

	cnts = cv.findContours(edged.copy(), cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	c = cnts[0]

	for x in cnts:
		if len(x) > len(c):
			c = x

	mask = np.zeros(gray.shape, dtype="uint8")
	cv.drawContours(mask, [c], -1, 255, -1)

	(x, y, w, h) = cv.boundingRect(c)
	imageROI = image[y:y + h, x:x + w]
	maskROI = mask[y:y + h, x:x + w]
	imageROI = cv.bitwise_and(imageROI, imageROI, mask=maskROI)


	bestImg = imageROI
	for angle in np.arange(0, 360, 3):
		rotated = crop_image(imutils.rotate_bound(imageROI, angle),0)
		if rotated.shape[0]/rotated.shape[1] > bestImg.shape[0]/bestImg.shape[1]:
			bestImg = rotated

	return cv.resize(bestImg, (800, 800), interpolation = cv.INTER_AREA)
