import cv2
import numpy as np
import config

from skimage.feature import hog

def extract_hog_features(img):
    return hog(img,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(3, 3),
               visualise=False,
               block_norm='L2-Hys')


def select_channel(img_rgb, channel):
    #pylint ignore-too-many-return
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

    def extract_region_of_interest(img):
        shape = np.shape(img)
        # make sure we have the expected size and that it is a single channel
        assert len(shape) == 2
        assert shape[0] == 720
        assert shape[1] == 1280

        row_start = config.SEARCH_REGION[0][1]
        row_end = config.SEARCH_REGION[1][1]
        col_start = config.SEARCH_REGION[0][0]
        col_end = config.SEARCH_REGION[1][0]

        return img[row_start:row_end, col_start:col_end, ]
