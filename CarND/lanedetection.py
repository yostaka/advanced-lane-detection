
import cv2
import matplotlib.pyplot as plt
import numpy as np

import CarND.perstransform as cntransform
import CarND.thresholding as cnthresh


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def getMaskedImage(img):
    # Mask the image to extract areas where lane line shows up
    # mask = np.zeros_like(img)
    # ignore_mask_color = 255

    # Define a four sided polygon to mask
    imshape = img.shape
    # vertices = np.array([[(0, imshape[0]), (imshape[1]*2/5, imshape[0]*3/5), (imshape[1]*3/5, imshape[0]*3/5), (imshape[1],imshape[0])]], dtype=np.int32)
    # vertices = np.array([[(160, imshape[0]), (570, 430), (680, 430), (1200, imshape[0])]], dtype=np.int32)
    vertices = np.array([[(160, imshape[0]), (590, 430), (680, 430), (1220, imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(img, vertices)
    return masked_edges


def getThresholdedBinaryImage(img, ksize=9, show_img=False):
    gradx = cnthresh.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = cnthresh.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = cnthresh.mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = cnthresh.dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    color_binary = cnthresh.color_threshold(img, thresh=(170, 255))

    combined = np.zeros_like(dir_binary, dtype='uint8')
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255
    # combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255

    masked_combined = getMaskedImage(combined)

    if show_img:
        f, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
        f.tight_layout()
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=40)
        axes[0, 1].imshow(gradx, cmap='gray')
        axes[0, 1].set_title('gradx', fontsize=40)
        axes[1, 0].imshow(grady, cmap='gray')
        axes[1, 0].set_title('grady', fontsize=40)
        axes[1, 1].imshow(mag_binary, cmap='gray')
        axes[1, 1].set_title('mag_binary', fontsize=40)
        axes[2, 0].imshow(dir_binary, cmap='gray')
        axes[2, 0].set_title('dir_binary', fontsize=40)
        axes[2, 1].imshow(color_binary, cmap='gray')
        axes[2, 1].set_title('color_binary', fontsize=40)
        axes[3, 0].imshow(combined, cmap='gray')
        axes[3, 0].set_title('Combined Thresholded Image', fontsize=40)
        axes[3, 1].imshow(masked_combined, cmap='gray')
        axes[3, 1].set_title('Masked Combined Image', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()

    return masked_combined


def getLaneMaskImage(img, show_img=False):
    warped_im = cntransform.warp(img)

    # Visualize undistortion
    if show_img:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Source image')
        ax1.imshow(img, cmap='gray')
        ax1.plot(761, 499, '.')
        ax1.plot(1034, 673, '.')
        ax1.plot(277, 673, '.')
        ax1.plot(528, 499, '.')
        ax2.set_title('Warped image')
        ax2.imshow(warped_im, cmap='gray')

        plt.show()

    # Detect lane pixels and fit to find the lane boundary
    histogram = np.sum(warped_im[warped_im.shape[0] // 2:, :], axis=0)

    if show_img:
        plt.plot(histogram)
        plt.show()

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_im, warped_im, warped_im))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lanes
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(warped_im.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_im.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100  # original was 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_im.shape[0] - (window + 1) * window_height
        win_y_high = warped_im.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynominal to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualize detected lane lines
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_im.shape[0] - 1, warped_im.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if show_img:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped_im, warped_im, warped_im)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right lane pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if show_img:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Fill area between left lane and right lane
    out_img = np.dstack((warped_im, warped_im, warped_im)) * 255
    window_img = np.zeros_like(out_img)
    detected_lane_img = np.zeros_like(out_img)

    # Color in left and right lane pixels
    detected_lane_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    detected_lane_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))[::-1]])
    lane_area_pts = np.hstack((left_line, right_line))

    # Fill the area between lanes onto warped blank image
    cv2.fillPoly(window_img, np.int_([lane_area_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, detected_lane_img, 1, 0)
    result = cv2.addWeighted(result, 1, window_img, 0.3, 0)
    if show_img:
        plt.imshow(result)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    unwarped_im = cntransform.unwarp(window_img)
    unwarped_lane_img = cntransform.unwarp(detected_lane_img)

    return unwarped_im, unwarped_lane_img


def getOverlayedImg(img, mtx, dist, show_img=False):

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    combined = getThresholdedBinaryImage(dst, show_img=show_img)
    area_img, lane_img = getLaneMaskImage(combined, show_img=show_img)

    out_img = cv2.addWeighted(img, 1, lane_img, 1, 0)
    out_img = cv2.addWeighted(out_img, 1, area_img, 0.3, 0)

    if show_img:
        plt.imshow(out_img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return out_img
