'''Preprocess module unit tests'''
import unittest

import matplotlib.image as mpimg
import numpy as np

import config
import preprocess as pp


class TestPreprocess(unittest.TestCase):
    """
    TestPreprocess unit tests.
        :param unittest.TestCase:
    """
    def test_extract_regions(self):
        """
        Test extracting scaled regions from the image.
            :param self:
        """
        img = mpimg.imread('test_images/test1.jpg')
        assert img is not None
        regions = pp.extract_regions_of_interest(img)
        self.assertTrue(len(regions) == config.SCALE_SAMPLES)
        for scaler, bounds, image in regions:
            self.assertEqual(np.shape(image)[0], 64)

    def test_hog_single_channel(self):
        """
        Test getting a hog layer from a single channel image.
            :param self:
        """
        dummy = np.zeros((64, 64, 1))

        hog = pp.extract_hog_features(dummy)
        self.assertGreater(len(hog), 2000)

    def test_hog_multi_channel(self):
        """
        Test getting a hog layer from a single channel image.
            :param self:
        """
        dummy = np.zeros((64, 64, 2))

        hog = pp.extract_hog_features(dummy)
        self.assertGreater(len(hog), 4000)

if __name__ == '__main__':
    unittest.main()
