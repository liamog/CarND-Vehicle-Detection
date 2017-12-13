'''Configuration values for vehicle detector'''
# Region within the image to search for cars.
NEAR_SEARCH_REGION = ((0, 392), (1280, 656))
FAR_SEARCH_REGION = ((10, 408), (1280 - 10, 460))
SCALE_SAMPLES = 8

#Which color channels to use for HOG feature extraction
INPUT_CHANNELS = ['hls_s']

#HOG parameters
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = 'L2-Hys'
HOG_BLOCK_STEPS = 3

#Other features to extract
USE_SPATIAL = True
USE_COLOR_HIST = True

#False Positive filtering
NUM_FRAMES_HEATMAP = 4
HEATMAP_THRESHOLD = 2

#Training search params for GridSearchCV
PARAM_GRID = [
    {'C': [1, 10, 100],
     'gamma': [0.01, 0.001, 0.0001],
     'kernel': ['linear']},
]

RESULTS_FOLDER = "results"
