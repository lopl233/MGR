import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import seaborn as sns
from scipy import misc
from scipy import ndimage
from scipy import signal


def togray(color_image):
    return cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)


def blur(sharp_image, blur_size):
    return cv.blur(sharp_image, (blur_size, blur_size))


def denoisegaus(noise_img, x):
    return ndimage.gaussian_filter(noise_img, x)

def denoisemedian(noise_img, x):
    return ndimage.median_filter(noise_img, x)

def resize(image, factor):
    return cv.resize(image, (0, 0), fx=factor, fy=factor)


def findcountour(image):
    ret, thresh = cv.threshold(image, 200, 255, 0)
    contours_list = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, )[0]
    blank_img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    contours_lengths = [len(x) for x in contours_list]
    contour = [x for x in contours_list if len(x) == max(contours_lengths)]
    coutour_img = cv.drawContours(blank_img, contour, -1, (255, 255, 255), 3)
    coutour_img = cv.cvtColor(coutour_img, cv.COLOR_BGR2GRAY)
    return cv.blur(coutour_img, (40, 40))


def applymask(image, mask):
    mask[mask > 0] = 1
    image = image * mask
    image[image == 0] = 255
    return image

def showimg(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def loadimg(number):
    return cv.imread('img/standardleaves/'+str(number)+'.jpg')

def plotheatmap(image):
    ax = sns.heatmap(image, center=140)
    plt.show()
    return

start = time.time()

org_image = togray(loadimg(3))
denoised_image = denoisemedian(org_image, 3)
denoised_image = signal.wiener(org_image, 7, 2)

#mask = findcountour(denoised_image)
#coutour_only = applymask(denoised_image, mask)
#plotheatmap(coutour_only)

denoised_image -= denoised_image.min()
denoised_image *= 255/denoised_image.max()
denoised_image = np.uint8(denoised_image)
print(denoised_image.min())
print(denoised_image.max())
print(denoised_image.shape)
print(denoised_image[555][555])

end = time.time()
print(end - start)

showimg(resize(denoised_image, 0.6))







