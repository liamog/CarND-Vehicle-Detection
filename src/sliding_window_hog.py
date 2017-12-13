import numpy as np
from skimage.feature import hog

import config
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import preprocess as pp

class SlidingWindowHog():

    def __init__(self):
        self.feature_windows = []
        self.visuals = None

    def process_image(self, img):
        '''Returns the set of feature vectors to classify from this subimage'''
        img_for_hog = pp.convert_img_for_hog(img)
        hogs_list, self.visuals = pp.extract_hog_features(img_for_hog, feature_vector=False)
        hogs = np.array(hogs_list)
        shape = np.shape(hogs)
        window_width = shape[1]
        img_width_in_blocks = shape[2]
        start_col = 0
        while (img_width_in_blocks - start_col >= window_width):
            hogs_subsample = hogs[:, :, start_col:start_col + window_width, :, :,]
            self.feature_windows.append((start_col, hogs_subsample.ravel()))
            start_col += config.HOG_BLOCK_STEPS
