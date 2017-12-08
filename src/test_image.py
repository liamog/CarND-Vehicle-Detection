"""Test lane lines."""

import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib
import cv2
from vehicle_detector import VehicleDetector

files = glob.glob('test_images/*.jpg')
files.sort()
pathlib.Path("output_images").mkdir(parents=True, exist_ok=True)

for file in files:
    # New dector for each test image as we don't want our heatmap to
    # cause weirdness over different frames.
    detector = VehicleDetector()
    print(file)
    name, ext = os.path.splitext(os.path.basename(file))
    target = os.path.join('output_images', name + '_detected.jpg')
    img_rgb = cv2.imread(file)
    final = detector.process_image_with_diagnostics(img_rgb)
    cv2.imwrite(target, final)
