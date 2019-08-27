import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import seaborn as sns
import functools
import operator
from scipy import misc
from scipy import ndimage
from functools import lru_cache
from scipy import stats
from rotate_pills import *
from scipy import signal
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


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


def findcountour(image, block , C):

    ret, thresh = cv.threshold(image, 240, 255, 0)
    #return  cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, C)
    #return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block, C)
    contours_list = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1, )[0]
    blank_img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    contours_lengths = [len(x) for x in contours_list]
    contour = [x for x in contours_list if len(x) == max(contours_lengths)]

    return contour[0]
    coutour_img = cv.drawContours(blank_img, contour, -1, (255, 255, 255), 3)
    coutour_img = cv.cvtColor(coutour_img, cv.COLOR_BGR2GRAY)
    return coutour_img


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
    ax = sns.heatmap(image)
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
    bumped_img = np.zeros((image.shape[0], image.shape[1]), np.uint16)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            bumped_img[x][y] = image[x][y] * bump_funcion(x, y, (image.shape[0]) / 2,
                                                                   (image.shape[1]) / 2, radius, width)
    return bumped_img

def generate_bump_images(width, height, bump_width ,start_radius, end_radius, step):
    radiuses = [x for x in range(start_radius,end_radius+1,step)]

    images = []

    for x in radiuses:
        b1 = np.zeros((width, height), np.uint32)
        b1[b1 == 0] = 1000
        images.append(bump_image(b1, x, bump_width))

    return images



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



def countObj(image):
    f = image.astype(np.uint8)
    ret, thresh = cv.threshold(f, 1, 255, 0)
    del f
    contours_list = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1, )[0]
    return len(contours_list)


def bumpedImageRetriveData(bumped_image, start_tresh, end_tresh, step):
    results = []
    for x in range(start_tresh, end_tresh, step):
        bumped_image[bumped_image < x] = 0
        results.append(countObj(bumped_image))

    return results


def generateData(org_image, bump_list):
    org_image = rotateUntilBest(org_image)
    org_image = cv.copyMakeBorder(org_image.copy(), 100, 100, 100, 100, cv.BORDER_CONSTANT, value=[0,0,0])
    org_image = denoisemedian(org_image, 3)
    org_image[org_image == 0] = 255

    KDEImage = getKDEImage(org_image)
    bumped_images = [x * KDEImage / 1000 for x in bump_list]

    return [bumpedImageRetriveData(x, 10, 200, 10) for x in bumped_images]


def getKDEImage(image):
    countour = findcountour(togray(image), 99, 10)

    X, Y = ([], [])

    xmin, xmax = 0, image.shape[1]
    ymin, ymax = 0, image.shape[0]

    X = [x[0][0] for x in countour]
    Y = [x[0][1] for x in countour]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([X, Y])
    kernel = stats.gaussian_kde(values, bw_method='scott')
    f = np.reshape(kernel(positions).T, xx.shape)
    f = f * (200 / f.max())
    return f





bump_list = generate_bump_images(300,300, 20, 20, 140, 6)
start = time.time()

data = generateData(loadimg(8), bump_list)

data2 = functools.reduce(operator.iconcat, data, [])
print(len(data2))
print(data2)
end = time.time()
print(end - start)






