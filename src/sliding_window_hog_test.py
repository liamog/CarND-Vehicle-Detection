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
        dummy = np.zeros((64,64))
        dummy_hog = pp.extract_hog_features(dummy, feature_vector=False)
        dummy_shape = np.shape(dummy_hog)

        img = np.zeros((64, 1280))

        # just test with one channel
        sliding_window = SlidingWindowHog()
        sliding_window.ProcessImage(img)
        for sub_image in sliding_window.images:
            self.assertEqual(np.shape(sub_image), dummy_shape)

if __name__ == '__main__':
    unittest.main()
