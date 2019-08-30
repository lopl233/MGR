import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import seaborn as sns
import functools
import operator
import pandas as pd
from scipy import misc
from scipy import ndimage
from functools import lru_cache
from scipy import stats
from rotate_pills import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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

    return [bumpedImageRetriveData(x, 10, 250, 4) for x in bumped_images]


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


def saveToCsv(Array, filename):
    df = pd.DataFrame(Array)
    export_csv = df.to_csv(r'C:\Users\Awangardowy Kaloryfe\Desktop\export_dataframe.csv', index=None, header=False)


def classify(number):
    if number >= 1001 and number <= 1059:
        return "pubescent bamboo"
    if number >= 1060 and number <= 1122:
        return "Chinese horse chestnut"
    if number >= 1552 and number <= 1616:
        return "Anhui Barberry"
    if number >= 1123 and number <= 1194:
        return "Chinese redbud"
    if number >= 1195 and number <= 1267:
        return "true indigo"
    if number >=1268  and number <= 1323:
        return "Japanese maple"
    if number >= 1324 and number <= 1385:
        return "Nanmu"
    if number >= 1386 and number <= 1437:
        return "castor aralia"
    if number >= 1497 and number <= 1551:
        return "Chinese cinnamon"
    if number >= 1438 and number <= 1496:
        return "goldenrain tree"
    if number >= 2001 and number <= 2050:
        return "Big-fruited Holly"
    if number >= 2051 and number <= 2113:
        return "Japanese cheesewood"
    if number >= 2114 and number <= 2165:
        return "wintersweet"
    if number >= 2166 and number <= 2230:
        return "camphortree"
    if number >= 2231 and number <= 2290:
        return "Japan Arrowwood"
    if number >= 2291 and number <= 2346:
        return "sweet osmanthus"
    if number >= 2347 and number <= 2423:
        return "deodar"
    if number >= 2424 and number <= 2485:
        return "ginkgo, maidenhair tree"
    if number >= 2486 and number <= 2546:
        return "Crape myrtle, Crepe myrtle"
    if number >= 2547 and number <= 2612:
        return "oleander"
    if number >= 2616 and number <= 2675:
        return "Sweet yew plum pine"
    if number >= 3001 and number <= 3055:
        return "Japanese Flowering Cherry"
    if number >= 3056 and number <= 3110:
        return "Glossy Privet"
    if number >= 3111 and number <= 3175:
        return "Chinese Toon"
    if number >= 3176 and number <= 3229:
        return "peach"
    if number >= 3230 and number <= 3281:
        return "Ford Woodlotus"
    if number >= 3282 and number <= 3334:
        return "trident maple"
    if number >= 3335 and number <= 3389:
        return "Beale's barberry"
    if number >= 3390 and number <= 3446:
        return "southern magnolia"
    if number >= 3447 and number <= 3510:
        return "Canadian poplar"
    if number >= 3511 and number <= 3563:
        return "Chinese tulip tree"
    if number >= 3566 and number <= 3621:
        return "tangerine"
    return "NA"


start = time.time()
bump_list = generate_bump_images(300,300, 20, 20, 140, 10)

print(time.time() - start)
start = time.time()


"""""
if number >= 1001 and number <= 1059:
    return "pubescent bamboo"
if number >= 1060 and number <= 1122:
"""""


trainingData = []
testData = []

labelsTraining = []
labelsTest = []
for x in range(1000,1020 + 1):
    type = classify(x)

    if type != 'NA':
        data = generateData(loadimg(x), bump_list)
        data2 = [functools.reduce(operator.iconcat, data, [])]
        trainingData.append(data2[0])
        labelsTraining.append(type)
        print(x)

for x in range(1060,1080 + 1):
    type = classify(x)

    if type != 'NA':
        data = generateData(loadimg(x), bump_list)
        data2 = [functools.reduce(operator.iconcat, data, [])]
        trainingData.append(data2[0])
        labelsTraining.append(type)
        print(x)

for x in range(1021, 1030 + 1):
    type = classify(x)

    if type != 'NA':
        data = generateData(loadimg(x), bump_list)
        data2 = [functools.reduce(operator.iconcat, data, [])]
        testData.append(data2[0])
        labelsTest.append(type)
        print(x)

for x in range(1081, 1100 + 1):
    type = classify(x)

    if type != 'NA':
        data = generateData(loadimg(x), bump_list)
        data2 = [functools.reduce(operator.iconcat, data, [])]
        testData.append(data2[0])
        labelsTest.append(type)
        print(x)



print(len(trainingData))
print(len(trainingData[0]))

print(time.time() - start)
start = time.time()


pca = PCA(n_components=5)
principalComponentsTrain = pca.fit_transform(trainingData)
principalComponentsTest = pca.fit(testData)

print(time.time() - start)
start = time.time()

logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(trainingData, labelsTraining)
predicted = logisticRegr.predict(testData)

print(time.time() - start)
start = time.time()

failed = 0
passed = 0
for x in range(0,len(predicted)):
    if predicted[x] == labelsTest[x]:
        passed = passed + 1
    else:
        failed = failed + 1

    print(predicted[x] == labelsTest[x], end ='')
    print(predicted[x], end = ' ')
    print(labelsTest[x])


print()
print(passed)
print(failed)
print(passed/(passed+failed)*100)




