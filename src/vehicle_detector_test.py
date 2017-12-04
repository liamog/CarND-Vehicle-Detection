'''vehicle_detector module unit tests'''
import unittest

import matplotlib.image as mpimg
import numpy as np

import config
import preprocess as pp
from vehicle_detector import VehicleDetector

class TestVehicleDetector(unittest.TestCase):
    """
    VehicleDetector unit tests.
        :param unittest.TestCase:
    """

    def test_sliding_windows(self):
        """
        Test extracting scaled regions from the image.
            :param self:
        """
        img = mpimg.imread('test_images/test1.jpg')

        # just test with one channel
        vehicle_detector = VehicleDetector()
        vehicle_detector.process_image(img)

if __name__ == '__main__':
    unittest.main()
