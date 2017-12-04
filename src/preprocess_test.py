import unittest
import os
import preprocess as pp
import config
import matplotlib.image as mpimg

class TestPreprocess(unittest.TestCase):

    def test_extract_regions(self):
        img = mpimg.imread('test_images/test1.jpg')
        assert (img is not None)
        # just test with one channel
        regions = pp.extract_regions_of_interest(img[:, :, 0])
        self.assertTrue(len(regions) == config.SCALE_SAMPLES)

if __name__ == '__main__':
    unittest.main()

