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

    def test_process_image(self):
        """
        Test a single image through the pipeline.
            :param self:
        """
        img = mpimg.imread('test_images/test1.jpg')
        self.assertIsNotNone(img)
        # just test with one channel
        vehicle_detector = VehicleDetector(unit_test=True)
        vehicle_detector.process_image(img)

    def test_process_diagnostics_image(self):
        """
        Test a single image through the pipeline.
            :param self:
        """
        img = mpimg.imread('test_images/test1.jpg')
        self.assertIsNotNone(img)
        # just test with one channel
        vehicle_detector = VehicleDetector(unit_test=True)
        vehicle_detector.process_image_with_diagnostics(img)

if __name__ == '__main__':
    unittest.main()
