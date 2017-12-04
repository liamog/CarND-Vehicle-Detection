'''Preprocess '''
import numpy as np
import scipy.misc
from skimage.feature import hog

import config
import cv2


def extract_hog_features(img, feature_vector=True):
    """
    Extract a set of hog features for the img.
        :param img:
    """
    shape = np.shape(img)
    assert len(shape) == 2
    #Only support extracting features for images px 64 high
    assert shape[0] == 64
    return hog(img,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(3, 3),
               visualise=False,
               feature_vector=feature_vector,
               block_norm='L2-Hys')

#pylint ignore-too-many-return
def select_channel(img_rgb, channel):
    """
    Returns the selected color sub channel from an RGB input image.
    First converts the colorspace if necessary, and then returns the
    requested channel
        :param img_rgb:
        :param channel: string describing the colorspace_channel e.g. hls_s
    """
    if channel == "rgb_r":
        return img_rgb[:, :, 0]
    if channel == "rgb_g":
        return img_rgb[:, :, 1]
    if channel == "rgb_b":
        return img_rgb[:, :, 2]
    if channel == "hls_h":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        return img_hls[:, :, 0]
    if channel == "hls_l":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        return img_hls[:, :, 1]
    if channel == "hls_s":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        return img_hls[:, :, 2]
    if channel == "yuv_y":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        return img_hls[:, :, 0]
    if channel == "yuv_u":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        return img_hls[:, :, 1]
    if channel == "yuv_v":
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        return img_hls[:, :, 2]
    if channel == "luv_y":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 0]
    if channel == "luv_u":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 1]
    if channel == "luv_v":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 2]

def extract_regions_of_interest(img):
    """
    Extracts a set of regions of interest for the image and scale them
    to 64px high.
        :param img:
    """
    shape = np.shape(img)
    # make sure we have the expected size and that it is a single channel
    assert len(shape) == 2
    print(shape)
    assert shape[0] == 720
    assert shape[1] == 1280

    row_far_start = config.FAR_SEARCH_REGION[0][1]
    row_far_end = config.FAR_SEARCH_REGION[1][1]
    col_far_start = config.FAR_SEARCH_REGION[0][0]
    col_far_end = config.FAR_SEARCH_REGION[1][0]

    row_near_start = config.NEAR_SEARCH_REGION[0][1]
    row_near_end = config.NEAR_SEARCH_REGION[1][1]
    col_near_start = config.NEAR_SEARCH_REGION[0][0]
    col_near_end = config.NEAR_SEARCH_REGION[1][0]

    row_start = row_near_start
    row_end = row_near_end
    col_start = col_near_start
    col_end = col_near_end

    row_start_step = int((row_near_start - row_far_start) / config.SCALE_SAMPLES)
    row_end_step = int((row_near_end - row_far_end) / config.SCALE_SAMPLES)
    col_start_step = int((col_near_start - col_far_start) / config.SCALE_SAMPLES)
    col_end_step = int((col_near_end - col_far_end) / config.SCALE_SAMPLES)

    regions = []

    #pylint: disable=unused-variable
    for sub_image_index in range(config.SCALE_SAMPLES):
        sub_img_bounds = ((col_start,row_start), (col_end, row_end))
        sub_img = img[row_start:row_end, col_start:col_end, ]
        # Now resize this image to be 64px high and scale appropriately.
        sub_image_shape = np.shape(sub_img)
        scaler = 64 / sub_image_shape[0]
        resize_shape = (64, int(sub_image_shape[1] * scaler))
        resized_subimage = scipy.misc.imresize(sub_img, resize_shape)
        regions.append((1.0/scaler, sub_img_bounds, resized_subimage))

        col_start -= col_start_step
        col_end -= col_end_step
        row_start -= row_start_step
        row_end -= row_end_step
    return regions
