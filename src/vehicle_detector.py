import math
import pathlib

import matplotlib.pyplot as pp
import numpy as np
import scipy.misc
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import find_objects, label
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure

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
        self.heatmap_diagnostics = None
        self.heatmap_diagnostics_2 = None
        self.heatmap_strict = None
        self.heatmap_filtered = None
        self.save_images_folder = save_images_folder
        self.count = 0
        self.bboxes = None
        if self.save_images_folder is not None:
            pathlib.Path(self.save_images_folder).mkdir(
                parents=True, exist_ok=True)

    def detect_peaks(self, image):
        neighborhood_size = 32
        data_max = maximum_filter(image, neighborhood_size)

        # self.heatmap_diagnostics_2 = np.copy(data_max)
        # self.heatmap_diagnostics_2 *= 10

        # maxima = (image == data_max)
        # diff = (data_max > config.HEATMAP_THRESHOLD_LOW)
        # maxima[diff == 0] = 0

        labels, num_objects = label(data_max)
        print("Number of maximums {}".format(num_objects))
        slices = find_objects(labels)
        centers = []
        for dy, dx in slices:
            x_center = int((dx.start + dx.stop - 1) / 2)
            y_center = int((dy.start + dy.stop - 1) / 2)
            centers.append((x_center, y_center))

        return centers


    def build_heat_map(self):
        shape = np.shape(self.detections_img)
        self.heatmap = np.zeros((shape[0], shape[1]), dtype=float)
        for detections_frame in self.detections:
            for box in detections_frame:
                self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1.0
        self.heatmap_diagnostics = np.copy(self.heatmap)
        self.heatmap_diagnostics *= 10
        self.heatmap_diagnostics_2 = np.zeros_like(self.final_img)
        # First filter out spurious detections picking a high threshold to
        # find the hottest centers.
        self.heatmap_strict = np.copy(self.heatmap)
        self.heatmap_strict[self.heatmap_strict <= config.HEATMAP_THRESHOLD_HIGH] = 0
        self.centers = self.detect_peaks(self.heatmap_strict)
        if len(self.centers) == 0: return
        # Now filter out based on mean and std deviation.
        # heat_non_zero_values = self.heatmap[self.heatmap.nonzero()]
        # heat_mean = heat_non_zero_values.mean()
        # heat_sigma = heat_non_zero_values.std()

        # heat_factor = 0.0
        # heatmap_strict = np.copy(self.heatmap)
        # heatmap_strict[self.heatmap <= heat_mean + (heat_sigma * heat_factor)] = 0

        # probable_detections = label(self.heatmap_strict)
        # grow each bounding box to a max size that could be the bbox of
        # a vehicle and then apply a lower threshold within this box
        # get a better estimate of the bbox of the vehicle.

        self.heatmap_filtered = np.zeros_like(self.heatmap_strict)
        SEARCH_WIDTH = 150
        SEARCH_HEIGHT = 100

        diag_image_dimension = int(math.ceil(math.sqrt(len(self.centers))))
        diag_image_rows, diag_image_cols, channels = (np.shape(self.final_img))
        diag_image_rows /= diag_image_dimension
        diag_image_cols /= diag_image_dimension
        diag_image_rows = int(diag_image_rows)
        diag_image_cols = int(diag_image_cols)

        counter = 0
        for col, row in self.centers:
            row_top = max(0, row - SEARCH_HEIGHT)
            row_bottom = min(719, row + SEARCH_HEIGHT)
            col_left = max(0, col - SEARCH_WIDTH)
            col_right = min(1279, col + SEARCH_WIDTH)
            self.heatmap_filtered[row_top:row_bottom, col_left:col_right] = \
                    self.heatmap[row_top:row_bottom, col_left:col_right]
            detected_image = self.final_img[row_top:row_bottom,
                                                 col_left:col_right, :]
            self.detected_images.append(detected_image)
            bounding_box, area = self.extract_bounding_box(detected_image)
            x, y, w, h = bounding_box
            if area > 1000:
                #probably a car , big enough mass.
                top_left = (col_left + x, row_top + y)
                bottom_right = (top_left[0] + w, top_left[1] + h)
                # Draw the box on the image
                cv2.rectangle(self.final_img,
                              top_left, bottom_right, (0, 0, 255), 6)

             #copy the image to diagnostic tile
            tile_row_start = int(counter / diag_image_dimension) * \
                diag_image_rows
            tile_col_start = int(counter % diag_image_dimension) * \
                diag_image_cols
            tile_row_end = int(tile_row_start + diag_image_rows)
            tile_col_end = int(tile_col_start + diag_image_cols)
            resized_tile = scipy.misc.imresize(
                detected_image, (diag_image_rows, diag_image_cols))
            self.heatmap_diagnostics_2[tile_row_start:tile_row_end,
                                       tile_col_start:tile_col_end, :] = resized_tile
            counter += 1

        self.heatmap_filtered[self.heatmap_filtered <=
                            config.HEATMAP_THRESHOLD_LOW] = 0

        self.labels = label(self.heatmap_filtered)


        # self.heatmap_diagnostics_2 *= 10
    def extract_bounding_box(self, subimage):
        detected_image = cv2.cvtColor(subimage, cv2.COLOR_BGR2HLS)
        img = detected_image[:, :, 2]
        ret, th1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(img, 30, 150)

        th1, contours, hierarchy = cv2.findContours(
            th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = max(contours, key=cv2.contourArea)
        print(cv2.contourArea(c))
        return (cv2.boundingRect(c), cv2.contourArea(c))

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
        # self.draw_labeled_bboxes(self.labels)
        self.draw_centers(self.centers)
        if self.save_images_folder is not None:
            filename = "{}/processed_image_{}.jpg".format(self.save_images_folder, self.count)
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
        hm = scipy.misc.imresize(self.heatmap_diagnostics, size)
        hm = np.dstack((hm, np.zeros_like(hm), np.zeros_like(hm)))
        hm2 = scipy.misc.imresize(self.heatmap_diagnostics_2, size)
        # hm2 = np.dstack((hm2, np.zeros_like(hm2), np.zeros_like(hm2)))

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
            filename = "{}/diag_image_{}.jpg".format(self.save_images_folder, self.count)
            cv2.imwrite(filename, cv2.cvtColor(
                final_plus_diags, cv2.COLOR_RGB2BGR))

        return final_plus_diags
