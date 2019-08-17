from rotate_pills import *
import cv2 as cv

def crop_image(img,tol=0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    return crop


def loadimg(number):
    return cv.imread('img/standardleaves/'+str(number)+'.jpg')

im = rotateUntilBest(loadimg(9))
imCrop = crop_image(im)
resizedImg = cv.resize(imCrop, (1000, 1000))
# Display cropped image
print("ratio")
print(imCrop.shape[0]/imCrop.shape[1])

resizedImg[resizedImg < 20] =  255

cv.imshow("Image", resizedImg)
cv.waitKey(0)