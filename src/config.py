'''Configuration values for vehicle detector'''

# Region within the image to search for cars.
NEAR_SEARCH_REGION = ((0, 392), (1280, 624))
FAR_SEARCH_REGION = ((10, 408), (1280 - 10, 460))
SCALE_SAMPLES = 10


HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (3, 3)
HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = 'L2-Hys'

HOG_COLOR_CHANNEL = 'hls_s'
