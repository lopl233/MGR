import matplotlib.pyplot as plt
import time
import math
import seaborn as sns
import functools
import operator
import pandas as pd
from scipy import misc
from scipy import ndimage
from functools import lru_cache
from scipy import stats
import numpy as np
import imutils
import cv2 as cv
import pickle


from sklearn.neighbors import KernelDensity

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
    ret, edged = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)

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

    return cv.resize(bestImg, (800, 800), interpolation = cv.INTER_AREA) , rotated.shape[0]/rotated.shape[1], len(c)


def kde2D(x, y, bandwidth, xbins=1j, ybins=1j, **kwargs):
    xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)
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
    contours_list = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1, )[0]
    contours_lengths = [len(x) for x in contours_list]
    contour = [x for x in contours_list if len(x) == max(contours_lengths)]
    return contour[0]


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
    org_image, ratio, coutourlen = rotateUntilBest(org_image)
    org_image = cv.copyMakeBorder(org_image.copy(), 100, 100, 100, 100, cv.BORDER_CONSTANT, value=[0,0,0])
    org_image = denoisemedian(org_image, 3)
    org_image[org_image == 0] = 255

    KDEImage = getKDEImage(org_image)
    bumped_images = [x * KDEImage / 1000 for x in bump_list]

    return [bumpedImageRetriveData(x, 5, 250, 2) for x in bumped_images], ratio, coutourlen


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


def getResult(proportion):

    #all 32 spieces
    complete_ranges = [
        [1001,1059],[1060,1122],[1552,1616],[1123,1194],[1195,1267],[1268,1323],[1324,1385],[1386,1437],
        [1497,1551],[1438,1496],[2001,2050],[2051,2113],[2114,2165],[2166,2230],[2231,2290],
        [2291,2346],[2347,2423],[2424,2485],[2486,2546],[2547,2612],[2616,2675],[3001,3055],[3056,3110],
        [3111,3175],[3176,3229],[3230,3281],[3282,3334],[3335,3389],[3390,3446],[3447,3510],[3511,3563],
        [3566,3621]
    ]


    #reduced set of spieces
    ranges = [
        [1001, 1059], [1123, 1194], [1195, 1267], [1268, 1323], [1324, 1385], [1386, 1437],
        [1438, 1496], [2001, 2050], [2051, 2113], [2114, 2165], [2166, 2230], [2231, 2290],
        [2291, 2346], [2347, 2423], [2486, 2546], [2616, 2675], [3056, 3110],
        [3282, 3334], [3335, 3389], [3390, 3446], [3447, 3510], [3511, 3563],
        [3566, 3621]
    ]

    testObserv = []
    testLables = []
    testclassicDims = []
    trainingObserv = []
    traininglables = []
    trainingclassicDims = []

    for x in ranges:

        for y in range(x[0], x[0]+ math.floor((x[1]-x[0]+1)*proportion)):
            type = classify(y)
            if type != 'NA':
                data, ratio, coutourlen = generateData(loadimg(y), bump_list)
                data2 = [functools.reduce(operator.iconcat, data, [])]
                trainingObserv.append(data2[0])
                traininglables.append(type)
                trainingclassicDims.append([ratio,coutourlen])
                print(y)
        for y in range(x[0]+ math.floor((x[1]-x[0]+1)*proportion), x[1] + 1):
            type = classify(y)
            if type != 'NA':
                data, ratio, coutourlen = generateData(loadimg(y), bump_list)
                data2 = [functools.reduce(operator.iconcat, data, [])]
                testObserv.append(data2[0])
                testLables.append(type)
                testclassicDims.append([ratio, coutourlen])
                print(y)

    return trainingObserv, traininglables, trainingclassicDims, testObserv, testLables, testclassicDims



start = time.time()
bump_list = generate_bump_images(300,300, 10, 10, 210, 6)

print(time.time() - start)
start = time.time()

#generate data
trainingData, labelsTraining, trainingClassicDims, testData, labelsTest, testClassicDims = getResult(0.85)


#save data
np.savetxt("trainingData",trainingData)
np.savetxt("testData",testData)

with open('labelsTraining', 'wb') as fp:
    pickle.dump(labelsTraining, fp)

with open('labelsTest', 'wb') as fp:
    pickle.dump(labelsTest, fp)

with open('testClassicDims', 'wb') as fp:
    pickle.dump(testClassicDims, fp)

with open('trainingClassicDims', 'wb') as fp:
    pickle.dump(trainingClassicDims, fp)

print(time.time() - start)
