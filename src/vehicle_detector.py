
import numpy as np
import scipy.misc
from scipy.ndimage.measurements import label
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
        self.classifier = Classifier()
        self.bounding_boxes = []
        self.detections = []
        self.final_img = None
        self.detections_img = None

    def build_heat_map(self):
        self.heatmap = np.zeros_like(self.detections_img)
        for detections_frame in self.detections:
            for box in detections_frame:
                self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        self.heatmap[self.heatmap <= config.HEATMAP_THRESHOLD] = 0

    def draw_labeled_bboxes(self, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(self.final_img, bbox[0], bbox[1], (0, 0, 255), 6)

    def process_image(self, img):
        # Extract each search region and resize to correct scale for search.
        self.detections_img = np.copy(img)
        self.final_img = np.copy(img)
        shape = np.shape(img)
        assert shape[0] == 720
        assert shape[1] == 1280
        self.detections.insert(0, [])
        if len(self.detections) > config.NUM_FRAMES_HEATMAP:
            del self.detections[-1]
        current_detections = self.detections[0]
        self.regions_of_interest = pp.extract_regions_of_interest(img)
        for scaler, bounds, scaled_region in self.regions_of_interest:
            sliding_window_hog = SlidingWindowHog()
            sliding_window_hog.process_image(scaled_region)

            for col_start, sub_features in sliding_window_hog.feature_windows:
                scaled_x = col_start * \
                    config.HOG_PIXELS_PER_CELL[0]

                # Grab the same region as the hog data to add the extra
                # color histogram and spatial features.
                img_subsample = scaled_region[:, scaled_x:scaled_x+64, :]
                other_features = pp.extract_other_features(img_subsample)
                features = np.concatenate(
                    (sub_features.ravel(), other_features))
                if self.classifier.predict(features) == 1:
                    orig_x = scaled_x * scaler

                    cv2.rectangle(scaled_region,
                                  (scaled_x, 0),
                                  (scaled_x + 64, 64),
                                  [0, 255, 0], 3)

                    start_x = int(orig_x)
                    end_x = start_x + int(64 * scaler)
                    start_y = bounds[0][1]
                    end_y = bounds[1][1]

                    print("found vehicle - block start = {}, scaled_px={}, orig_px={}, bounds={}".format(
                        col_start,
                        scaled_x,
                        orig_x,
                        ((start_x, start_y),(end_x, end_y)),
                    ))
                    current_detections.append(
                        ((start_x, start_y), (end_x, end_y)))
                    cv2.rectangle(self.detections_img,
                                  (start_x, start_y),
                                  (end_x , end_y),
                                  [0,255,0], 3)

        # TODO Build Heatmap
        self.build_heat_map()
        # TODO Extract labels from heatmap
        labels = label(self.heatmap)

        # TODO Draw the bounding box on the image.
        self.draw_labeled_bboxes(labels)
        return self.final_img

    def process_image_with_diagnostics(self, img):
        processed = self.process_image(img)
        """Process the image and append diagnostics to image."""
        size = np.shape(img)

        # 4 panels of diagnostics
        size = (int(size[0] / 2), int(size[1] / 2))

        # raw detection boxes
        det = scipy.misc.imresize(self.detections_img, size)

        # Heat map
        hm = scipy.misc.imresize(self.heatmap, size)

        # Search regions
        vsize = int(size[0]/len(self.regions_of_interest))
        region_size=(vsize, size[1])
        slice_imgs = []
        for scaler, bounds, scaled_region in self.regions_of_interest:
            rii = scipy.misc.imresize(scaled_region, region_size)
            slice_imgs.append(rii)
        regions = np.vstack(slice_imgs)

        diags_1_r1 =np.hstack((det, hm))
        diags_1_r2 = np.hstack((regions, np.zeros_like(regions)))
        diags_1 = np.vstack((diags_1_r1, diags_1_r2))

        final_plus_diags = np.hstack((processed, diags_1))
        return final_plus_diags
