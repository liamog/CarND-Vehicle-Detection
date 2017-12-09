import numpy as np
import scipy
import scipy.misc
from moviepy.editor import VideoFileClip
from scipy.ndimage.interpolation import zoom

import cv2
from vehicle_detector import VehicleDetector

diagnostics_enabled = False
regular_enabled = False
trouble_1 = True
# input_base = "harder_challenge_video"
# input_base = "challenge_video"
input_base = "project_video"

input_filename = input_base + ".mp4"
output_filename = input_base + "_with_vehicles.mp4"
output_diag_filename_t1 = input_base + "_t1_with_vehicles.mp4"
output_diag_filename = input_base + "_with_diagnostics.mp4"
detector = VehicleDetector()

if trouble_1:
    count = 0
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(
        detector.process_image_with_diagnostics).subclip(38, 42)
    clip.write_videofile(output_diag_filename_t1, audio=False)

if regular_enabled:
    count = 0
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(detector.process_image)
    clip.write_videofile(output_filename, audio=False)

if diagnostics_enabled:
    count = 0
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(detector.process_image_with_diagnostics)
    clip.write_videofile(output_diag_filename, audio=False)
