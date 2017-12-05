
import numpy as np
from skimage.feature import hog

import config
import cv2
import preprocess as pp
from classifier import Classifier
from sliding_window_hog import SlidingWindowHog


class VehicleDetector():
    """The Vehicle Detector Class."""

    def __init__(self):
            """Initializer."""
            # Set to the image currently being processed
            self.source_img = None
            self.classifier = Classifier()

    def process_image(self, img):
        # Extract each search region and resize to correct scale for search.
        shape = np.shape(img)
        assert shape[0] == 720
        assert shape[1] == 1280
        for scaler, bounds, scaled_region in pp.extract_regions_of_interest(img):
            sliding_window_hog = SlidingWindowHog()
            sliding_window_hog.process_image(scaled_region)
            for col_start, sub_features in sliding_window_hog.feature_windows:
                if self.classifier.predict(sub_features.ravel()) == 1:
                    scaled_x = col_start * \
                        config.HOG_PIXELS_PER_CELL[0]
                    orig_x = scaled_x * scaler

                    start_x = int(orig_x)
                    end_x = start_x + int(64 * scaler)
                    start_y = bounds[0][1]
                    end_y = bounds[1][1]

                    print("found vehicle - block start = {}, scaled_px={}, orig_px={}, bounds={}".format(
                        col_start,
                        scaled_x,
                        orig_x,
                        bounds
                        ))
                    # TODO Draw the bounding box on the image.
                    cv2.rectangle(img,
                                  (start_x, start_y),
                                  (end_x , end_y),
                                  [0,255,0], 3)

        return img

    def process_image_with_diagnostics(self, img):
        processed = self.process_image(img)


