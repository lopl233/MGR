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


def bump_funcion(x, y ,centerx, centery, radius, width):
    dist = np.sqrt(np.power(x - centerx, 2) + np.power(y - centery, 2))
    pow = - np.power(dist-radius, 2) / (2 * width)
    return np.e.__pow__(pow)


print(bump_funcion(0, 5, 5, 5, 5, 1.5))
print(bump_funcion(0, 5, 5, 5, 4.5, 1.5))
print(bump_funcion(0, 5, 5, 5, 4, 1.5))
print(bump_funcion(0, 5, 5, 5, 3.5, 1.5))
print(bump_funcion(0, 5, 5, 5, 3, 1.5))
print(bump_funcion(0, 5, 5, 5, 2.5, 1.5))
print(bump_funcion(0, 5, 5, 5, 2, 1.5))
print(bump_funcion(0, 5, 5, 5, 1.5, 1.5))
print(bump_funcion(0, 5, 5, 5, 1, 1.5))
print(bump_funcion(0, 5, 5, 5, 0.5, 1.5))
print(bump_funcion(0, 5, 5, 5, 0, 1.5))