import pathlib

import matplotlib.pyplot as pp
import numpy as np
import scipy.misc
from scipy.ndimage.measurements import label

import config
import cv2
import preprocess as pp
from classifier import Classifier
from sliding_window_hog import SlidingWindowHog


class VehicleDetector():
    """The Vehicle Detector Class."""

    def __init__(self, unit_test=False, save_images_folder=None):
        """Initializer."""
        self.classifier = Classifier(unit_test)
        self.bounding_boxes = []
        self.detections = []
        self.final_img = None
        self.detections_img = None
        self.heatmap = None
        self.detected_images = []
        self.heatmap_debug = None
        self.filtered_heatmap_debug = None
        self.filtered_heatmap = None
        self.detected_vehicles = None
        self.save_images_folder = save_images_folder
        self.count = 0
        self.bboxes = None
        if self.save_images_folder is not None:
            pathlib.Path(self.save_images_folder).mkdir(
                parents=True, exist_ok=True)
            pathlib.Path(self.save_images_folder + "/detections").mkdir(
                parents=True, exist_ok=True)

    def build_heat_map(self):
        shape = np.shape(self.detections_img)

        self.heatmap = np.zeros((shape[0], shape[1]), dtype=float)
        self.heatmap_debug = np.copy(self.heatmap)
        self.filtered_heatmap_debug = np.copy(self.heatmap)
        num_frames_integrated = len(self.detections)
        print(num_frames_integrated)

        for detections_frame in self.detections:
            frame_heatmap = np.zeros_like(self.heatmap)
            for box in detections_frame:
                # Only count a pixel once for a full frame of video , even if we
                # get multiple hits from different scales.
                frame_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] = 1.0
            # Sum each pixel for each frame.
            self.heatmap += frame_heatmap

        self.filtered_heatmap = np.copy(self.heatmap)

        # Apply heatmap factor filter. First get the percentage
        self.filtered_heatmap = self.filtered_heatmap / num_frames_integrated

        self.filtered_heatmap[self.filtered_heatmap <=
                            config.HEATMAP_THRESHOLD_FACTOR] = 0

        self.heatmap_debug = np.copy(self.heatmap)
        self.filtered_heatmap_debug = np.copy(self.filtered_heatmap)

        # skip detection if we don't have a minimum number of frames first.
        # do this here so we still get diagnostic images.
        if num_frames_integrated < config.MIN_FRAMES:
            return
        self.detected_vehicles = label(self.filtered_heatmap)

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
            cv2.rectangle(self.final_img, bbox[0], bbox[1], (255, 0, 255), 6)

    def draw_centers(self, centers, radius=6) :
        for center in centers:
            cv2.circle(self.final_img, center, radius, (255,0,0),3)

    def get_centers(self, labels):
        centers = []
            # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            half_width = int((np.max(nonzerox) - np.min(nonzerox)) / 2)
            half_height = int((np.max(nonzeroy) - np.min(nonzeroy)) / 2)
            center = (np.min(nonzerox) + half_width, np.min(nonzeroy) + half_height)
            centers.append(center)
        return centers

    def process_image(self, img):
        self.count += 1

        # Extract each search region and resize to correct scale for search.
        self.detections_img = np.copy(img)
        self.final_img = np.copy(img)
        shape = np.shape(img)
        assert shape[0] == 720
        assert shape[1] == 1280

        # Prune older video frames from heatmap
        self.detections.insert(0, [])
        if len(self.detections) > config.NUM_FRAMES_HEATMAP:
            del self.detections[-1]

        # Fill in current detections
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
                other_features = []
                if config.USE_SPATIAL or config.USE_COLOR_HIST:
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

                    current_detections.append(
                        ((start_x, start_y), (end_x, end_y)))
                    cv2.rectangle(self.detections_img,
                                  (start_x, start_y),
                                  (end_x , end_y),
                                  [0,255,0], 3)

        self.build_heat_map()
        if self.detected_vehicles:
            self.draw_labeled_bboxes(self.detected_vehicles)

        if self.save_images_folder is not None:
            filename = "{}/processed_image_{}.jpg".format(
                self.save_images_folder, self.count)
            cv2.imwrite(filename, cv2.cvtColor(
                self.final_img, cv2.COLOR_RGB2BGR))
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
        hm = scipy.misc.imresize(self.heatmap_debug, size)
        hm = np.dstack((hm, np.zeros_like(hm), np.zeros_like(hm)))
        hm2 = scipy.misc.imresize(self.filtered_heatmap_debug, size)
        hm2 = np.dstack((hm2, np.zeros_like(hm2), np.zeros_like(hm2)))

        # Search regions
        vsize = int(size[0]/len(self.regions_of_interest))
        region_size=(vsize, size[1])
        slice_imgs = []
        for scaler, bounds, scaled_region in self.regions_of_interest:
            rii = scipy.misc.imresize(scaled_region, region_size)
            slice_imgs.append(rii)
        regions = np.vstack(slice_imgs)

        diags_1_r1 =np.hstack((det, hm))
        diags_1_r2 = np.hstack((regions, hm2))
        diags_1 = np.vstack((diags_1_r1, diags_1_r2))

        final_plus_diags = np.hstack((processed, diags_1))
        if self.save_images_folder is not None:
            filename = "{}/detections/det_image_{}.jpg".format(
                self.save_images_folder, self.count)
            cv2.imwrite(filename, cv2.cvtColor(
                det, cv2.COLOR_RGB2BGR))

            filename = "{}/diag_image_{}.jpg".format(
                self.save_images_folder, self.count)
            cv2.imwrite(filename, cv2.cvtColor(
                final_plus_diags, cv2.COLOR_RGB2BGR))

        return final_plus_diags
