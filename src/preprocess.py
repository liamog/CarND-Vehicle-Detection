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

def extract_regions_of_interest(img):
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

    # cv2.rectangle(img,
    #             (col_near_start, row_near_start),
    #             (col_near_end, row_near_end),
    #             [255, 0, 0], 3)

    # cv2.rectangle(img,
    #             (col_far_start, row_far_start),
    #             (col_far_end, row_far_end),
    #             [0, 0, 255], 3)


    row_start = row_near_start
    row_end = row_near_end
    col_start = col_near_start
    col_end = col_near_end

    row_start_step = int((row_near_start - row_far_start) / config.SCALE_SAMPLES)
    row_end_step = int((row_near_end - row_far_end) / config.SCALE_SAMPLES)
    col_start_step = int((col_near_start - col_far_start) / config.SCALE_SAMPLES)
    col_end_step = int((col_near_end - col_far_end) / config.SCALE_SAMPLES)

# # for file in glob.glob('test_images/test4.jpg'):
# print("row_start_step={}, row_end_step={}, col_start_step={}, col_end_step={}".format(
#     row_start_step, row_end_step, col_start_step, col_end_step))
    regions = []

    for ii in range(config.SCALE_SAMPLES):
        # print("r({}-{}), c({}-{})".format(row_start, row_end, col_start, col_end))
        regions.append(img[row_start:row_end, col_start:col_end, ])

        # cv2.rectangle(img, (col_start, row_start),
        #             (col_end, row_end), [0, 255, 0], 3)
        col_start -= col_start_step
        col_end -= col_end_step
        row_start -= row_start_step
        row_end -= row_end_step
    return regions
