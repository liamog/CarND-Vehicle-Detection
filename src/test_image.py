"""Test lane lines."""

import glob
import os
import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config
import cv2
from vehicle_detector import VehicleDetector

files = glob.glob('test_images_conseq/*.jpg')
files.sort()

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

configs = []

configs.append({
    "INPUT_CHANNELS": ['hls_s'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": False,
    "USE_COLOR_HIST": False,
    "RESULTS_FOLDER": "results/hls_s_no_extra_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['hls_h', 'hls_l', 'hls_s'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/hls_hls_spatial_color_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['luv_l'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": False,
    "USE_COLOR_HIST": False,
    "RESULTS_FOLDER": "results/luv_l_no_extra_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['luv_l'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/luv_l_color_spatial_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['luv_u'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/luv_u_color_spatial_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['luv_v'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/luv_v_color_spatial_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['luv_l', 'luv_v'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/luv_lv_color_spatial_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": False,
    "USE_COLOR_HIST": False,
    "RESULTS_FOLDER": "results/yuv_yuv_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['yuv_u', 'yuv_v'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": False,
    "USE_COLOR_HIST": False,
    "RESULTS_FOLDER": "results/yuv_uv_3x3_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
    "HOG_CELLS_PER_BLOCK": (2, 2),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": False,
    "USE_COLOR_HIST": False,
    "RESULTS_FOLDER": "results/yuv_yuv_2x2_steps_4"
})

configs.append({
    "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
    "HOG_CELLS_PER_BLOCK": (3, 3),
    "HOG_BLOCK_STEPS": 4,
    "USE_SPATIAL": True,
    "USE_COLOR_HIST": True,
    "RESULTS_FOLDER": "results/yuv_yuv_3x3_color_spatial_steps_4"
})

for c in configs:
    override_config_from_dict(c)
    target_dir = os.path.join(config.RESULTS_FOLDER, 'output_images')
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector()
    for file in files:
        # New dector for each test image as we don't want our heatmap to
        # cause weirdness over different frames.
        print(file)
        name, ext = os.path.splitext(os.path.basename(file))

        target = os.path.join(target_dir, name + '_detected.jpg')
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final = detector.process_image_with_diagnostics(img_rgb)
        img_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        cv2.imwrite(target, img_rgb)
