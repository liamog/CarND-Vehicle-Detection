import numpy as np
import scipy
import scipy.misc
from moviepy.editor import VideoFileClip
from scipy.ndimage.interpolation import zoom

import config
import cv2
from vehicle_detector import VehicleDetector

def override_config_from_dict(input_values):
    if input_values["INPUT_CHANNELS"] is not None:
        config.INPUT_CHANNELS = input_values["INPUT_CHANNELS"]

    if input_values["HOG_CELLS_PER_BLOCK"] is not None:
        config.HOG_CELLS_PER_BLOCK = input_values["HOG_CELLS_PER_BLOCK"]

    if input_values["HOG_BLOCK_STEPS"] is not None:
        config.HOG_BLOCK_STEPS = input_values["HOG_BLOCK_STEPS"]

    if input_values["USE_SPATIAL"] is not None:
        config.USE_SPATIAL = input_values["USE_SPATIAL"]

    if input_values["USE_COLOR_HIST"] is not None:
        config.USE_COLOR_HIST = input_values["USE_COLOR_HIST"]

    if input_values["RESULTS_FOLDER"] is not None:
        config.RESULTS_FOLDER = input_values["RESULTS_FOLDER"]

    if input_values["NUM_FRAMES_HEATMAP"] is not None:
        config.NUM_FRAMES_HEATMAP = input_values["NUM_FRAMES_HEATMAP"]

    if input_values["HEATMAP_THRESHOLD"] is not None:
        config.HEATMAP_THRESHOLD = input_values["HEATMAP_THRESHOLD"]

override_config_from_dict({
    "INPUT_CHANNELS": ['hls_h', 'hls_l', 'hls_s'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/hls_hls_spatial_color_3x3_steps_4",
    "NUM_FRAMES_HEATMAP": 25,
    "HEATMAP_THRESHOLD": 3
})

diagnostics_enabled = False
regular_enabled = False
trouble_1 = True
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
        detector.process_image_with_diagnostics).subclip(35, 45)
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
