'''Configuration values for vehicle detector'''

# Region within the image to search for cars.
NEAR_SEARCH_REGION = ((0, 392), (1280, 624))
FAR_SEARCH_REGION = ((10, 408), (1280 - 10, 460))
SCALE_SAMPLES = 5

INPUT_CHANNELS = ['rgb_r', 'rgb_g', 'rgb_b']

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = 'L2-Hys'

USE_SPATIAL = True
USE_COLOR_HIST = True

NUM_FRAMES_HEATMAP = 4
HEATMAP_THRESHOLD = 2

PARAM_GRID = [
    {'C': [1, 10, 100],
     'gamma': [0.01, 0.001, 0.0001],
     'kernel': ['linear']},
]
