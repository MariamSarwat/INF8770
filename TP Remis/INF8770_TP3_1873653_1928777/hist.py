import cv2 as cv
from scipy.spatial import distance
import numpy as np

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html

BIN_COUNT = [8,8,8]
RANGE = [0, 256, 0, 256, 0, 256]

# histogramme différent vérifier originale
def get_image_histograms(image):
    hist = cv.calcHist([image], [0,1,2], None, BIN_COUNT, RANGE)
    hist = cv.normalize(hist, hist).flatten()
    return hist

def get_chebyshev_distance(histsA, histsB):
    return distance.chebyshev(histsA, histsB)

def get_euclidean_distance(histsA, histsB):
    return distance.euclidean(histsA, histsB)
