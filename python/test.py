import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import seaborn as sns
from scipy import misc
from scipy import ndimage
from functools import lru_cache
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

def plotheatmap(image, centr):
    ax = sns.heatmap(image, center=centr)
    #ax = sns.heatmap(image)
    plt.show()
    return

def surface_plot (matrix):
    lena = misc.imresize(matrix, 0.08, interp='cubic')
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
    pow = - np.power(dist-radius, 2) / (2 * np.power(width,2))
    return np.power(np.e, pow)

@lru_cache(maxsize=999999999)
def gaussian_density_estimator_part(xval, yval, bandwidth, n):
    pow = -0.5 * np.power(((xval-yval)/bandwidth), 2)
    return np.power(np.e, pow)/n


def bump_image(image, radius, width):
    bumped_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for x in range(denoised_image.shape[0]):
        for y in range(denoised_image.shape[1]):
            bumped_img[x][y] = denoised_image[x][y] * bump_funcion(x, y, (image.shape[0]) / 2,
                                                                   (image.shape[1]) / 2, radius, width)
    return bumped_img


def rescale_matrix(matrix):
    matrix = matrix - matrix.min()
    return matrix.max() - matrix


def kernel_for_pixel(matrix, x, y, kernel_size):
    x_start = max(x - kernel_size, 0)
    x_end = min(x + kernel_size + 1, matrix.shape[0])

    y_start = max(y - kernel_size, 0)
    y_end = min(y + kernel_size +1, matrix.shape[1])

    return matrix[x_start:x_end, y_start:y_end]


def compute_gde_single_pixel(matrix, x, y, kernel_size, bandwidth):
    kernel = kernel_for_pixel(matrix, x, y, kernel_size)
    kernel = kernel.ravel()
    size = len(kernel)
    x_val = matrix[x][y]
    return sum([gaussian_density_estimator_part(x_val, x, bandwidth, size) for x in kernel])


def gde(matrix, kernel_size, bandwidth):
    gde_image = np.zeros((matrix.shape[0], matrix.shape[1]))

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            print(x, y)
            gde_image[x][y] = compute_gde_single_pixel(matrix, x, y , kernel_size, bandwidth)
    return gde_image


start = time.time()

org_image = togray(loadimg(2))
org_image = resize(org_image, 0.2)

#denoised_image = signal.wiener(org_image, noise=3)
#denoised_image = rescale_matrix(denoised_image)
denoised_image = denoisemedian(org_image, 2)

#bumped_image = bump_image(denoised_image, 400, 35)

denoised_image = 1 -denoised_image
denoised_image[denoised_image > 0] += 600
gde_img = gde(denoised_image, 19, 20)
gde_img = 1 - gde_img
gde_img = gde_img * 100000
gde_img = gde_img - 75000
gde_img[gde_img < 0] = 0
plotheatmap(gde_img, 15000)



end = time.time()
print(end - start)








