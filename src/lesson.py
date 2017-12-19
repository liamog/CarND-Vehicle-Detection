
    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
                imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hsta
# configs.append({
#     "INPUT_CHANNELS": ['hls_s'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": False,
#     "USE_COLOR_HIST": False,
#     "RESULTS_FOLDER": "results/hls_s_no_extra_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['hls_h', 'hls_l', 'hls_s'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/hls_hls_spatial_color_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['luv_l'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": False,
#     "USE_COLOR_HIST": False,
#     "RESULTS_FOLDER": "results/luv_l_no_extra_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['luv_l'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/luv_l_color_spatial_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['luv_u'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/luv_u_color_spatial_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['luv_v'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/luv_v_color_spatial_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['luv_l', 'luv_v'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/luv_lv_color_spatial_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": False,
#     "USE_COLOR_HIST": False,
#     "RESULTS_FOLDER": "results/yuv_yuv_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['yuv_u', 'yuv_v'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": False,
#     "USE_COLOR_HIST": False,
#     "RESULTS_FOLDER": "results/yuv_uv_3x3_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
#     "HOG_CELLS_PER_BLOCK": (2, 2),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": False,
#     "USE_COLOR_HIST": False,
#     "RESULTS_FOLDER": "results/yuv_yuv_2x2_steps_4"
# })

# configs.append({
#     "INPUT_CHANNELS": ['yuv_y', 'yuv_u', 'yuv_v'],
#     "HOG_CELLS_PER_BLOCK": (3, 3),
#     "HOG_BLOCK_STEPS": 4,
#     "USE_SPATIAL": True,
#     "USE_COLOR_HIST": True,
#     "RESULTS_FOLDER": "results/yuv_yuv_3x3_color_spatial_steps_4"
# })
ck((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left +
                                                                              win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        return draw_img

    ystart = 400
    ystop = 656
    scale = 1.5

    out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    plt.imshow(out_img)
