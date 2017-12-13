'''sliding_window module unit tests'''
import unittest

import matplotlib.image as mpimg
import numpy as np

import config
import preprocess as pp
from sliding_window_hog import SlidingWindowHog

class TestSlidingWindow(unittest.TestCase):
    """
    SlidingWindow unit tests.
        :param unittest.TestCase:
    """

    def test_sliding_windows(self):
        """
        Test extracting scaled regions from the image.
            :param self:
        """
        dummy = np.zeros((64, 64, 3), np.uint8)
        img_for_hog = pp.convert_img_for_hog(dummy)
        dummy_hog, dummy_visuals = pp.extract_hog_features(img_for_hog)
        dummy_shape = np.shape(dummy_hog)

        img = np.zeros((64, 1280, 3), np.uint8)
        # just test with one channel
        sliding_window = SlidingWindowHog()
        sliding_window.process_image(img)
        for start_col, sub_image in sliding_window.feature_windows:
            self.assertEqual(np.shape(sub_image), dummy_shape)

if __name__ == '__main__':
    unittest.main()
