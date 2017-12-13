'''Preprocess '''
import numpy as np
import scipy.misc
from skimage.feature import hog

import config
import cv2

def convert_img_for_hog(img_rgb):
    """
    Converts the image to the set of color channels specified in config.
        :param img_rgb: image to convert in RGB
    """
    channels = []
    for channel_name in config.INPUT_CHANNELS:
        channel = select_channel(img_rgb, channel_name)
        channel = channel / 255
        channels.append(channel)
    return np.stack(channels, axis=-1)

def extract_hog_features(img, feature_vector=True):
    """
    Extract a set of hog features for the img.
        :param img: image to extract
        :param feature_vector: image to extract
    """
    shape = np.shape(img)
    #Only support extracting features for images px 64 high but any width
    #For single channel images use a 64xNx1 array.
    #For multiple channel images use a 64xNxM array.
    assert len(shape) == 3
    assert shape[0] == 64
    hogs = []
    visuals = []
    for ii in range(shape[2]):
        hog_feat, vis = hog(img[:, :, ii],
                orientations=config.HOG_ORIENTATIONS,
                pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                cells_per_block=config.HOG_CELLS_PER_BLOCK,
                visualise=True,
                feature_vector=False,
                block_norm='L2-Hys')
        hogs.append(hog_feat)
        visuals.append(hog_feat)

    if feature_vector:
        return np.ravel(hogs), visuals

    return hogs, visuals

def bin_spatial(img, size=(16, 16)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=16):
    # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def extract_other_features(img):
    retval = None
    if config.USE_SPATIAL:
       retval = bin_spatial(img)
    if config.USE_COLOR_HIST:
        colors = color_hist(img)
        if retval is not None:
            np.concatenate((retval, colors))
        else :
            retval = colors
    return retval

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

    if channel == "luv_l":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 0]
    if channel == "luv_u":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 1]
    if channel == "luv_v":
        img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        return img_luv[:, :, 2]

    if channel == "hsv_h":
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        return img_hsv[:, :, 0]
    if channel == "hsv_s":
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        return img_hsv[:, :, 1]
    if channel == "hsv_v":
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        return img_hsv[:, :, 2]

    if channel == "ycrcb_y":
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        return img_ycrcb[:, :, 0]
    if channel == "ycrcb_cr":
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        return img_ycrcb[:, :, 1]
    if channel == "ycrcb_cb":
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        return img_ycrcb[:, :, 2]

def extract_regions_of_interest(img):
    """
    Extracts a set of regions of interest for the image and scale them
    to 64px high.
        :param img:
    """
    shape = np.shape(img)
    # make sure we have the expected size.
    assert len(shape) == 3
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
        sub_img = img[row_start:row_end, col_start:col_end, :]
        # Now resize this image to be 64px high and scale appropriately.
        sub_image_shape = np.shape(sub_img)
        scaler = 64 / sub_image_shape[0]
        resize_shape = (64, int(sub_image_shape[1] * scaler), shape[2])
        resized_subimage = scipy.misc.imresize(sub_img, resize_shape)
        regions.append((1.0/scaler, sub_img_bounds, resized_subimage))

        col_start -= col_start_step
        col_end -= col_end_step
        row_start -= row_start_step
        row_end -= row_end_step
    return regions

