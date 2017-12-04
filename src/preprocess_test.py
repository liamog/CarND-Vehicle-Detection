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
        # just test with one channel
        regions = pp.extract_regions_of_interest(img[:, :, 0])
        self.assertTrue(len(regions) == config.SCALE_SAMPLES)
        for image in regions:
            self.assertEqual(np.shape(image)[0], 64)

if __name__ == '__main__':
    unittest.main()
