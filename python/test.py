import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import seaborn as sns
from scipy import misc
from scipy import ndimage
from scipy import signal
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


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
    return cv.blur(coutour_img, (30, 30))


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
    ax = sns.heatmap(image, center=90)
    plt.show()
    return

def surface_plot (matrix):
    lena = misc.imresize(coutour_only, 0.08, interp='cubic')
    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, lena, rstride=1, cstride=1, cmap=plt.cm.jet,
                    linewidth=0)
    # show it
    plt.show()


def bump_funcion(x, y ,centerx, centery, radius, width):
    dist = np.sqrt(np.power(x - centerx, 2) + np.power(y - centery, 2))
    pow = - np.power(dist-radius, 2) / (2 * width)
    return np.e.__pow__(pow)


def gaussian_density_estimator_part(xval, yval, bandwidth, n):
    pow = -0.5 * np.power(((xval-yval)/bandwidth), 2)
    return np.e.__pow__(pow)/n


start = time.time()

org_image = togray(loadimg(6))
#denoised_image = denoisemedian(org_image, 3)
denoised_image = signal.wiener(org_image,9,3)

mask = findcountour(org_image)
coutour_only = applymask(denoised_image, mask)
coutour_only = coutour_only * -1
coutour_only = coutour_only - coutour_only.min()

surface_plot(coutour_only)
#plotheatmap(coutour_only)


end = time.time()
print(end - start)


#showimg(resize(denoised_image, 0.6))







