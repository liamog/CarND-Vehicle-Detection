import numpy as np
from skimage.feature import hog

import config
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

class SlidingWindowHog():

    def __init__(self, horizontal_block_step = 2):
        self.images = []
        self.hog_samples = None
        self.horizontal_block_step = horizontal_block_step

    def ProcessImage(self, img):
        '''Returns the set of feature vectors to classify from this subimage'''
        # process the hog image as a set of feature blocks that we will
        # subsample
        self.hog_samples = hog(img,
                               orientations=config.HOG_ORIENTATIONS,
                               pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                               cells_per_block=config.HOG_CELLS_PER_BLOCK,
                               block_norm=config.HOG_BLOCK_NORM,
                               feature_vector=False)


        shape = np.shape(self.hog_samples)
        print(shape)
        window_width = shape[0]

        start_col = 0
        while (shape[1] - start_col >= window_width):
            self.images.append(
                self.hog_samples[:, start_col:start_col + window_width, :, :, ])
            start_col += self.horizontal_block_step
