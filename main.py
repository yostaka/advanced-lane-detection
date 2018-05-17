import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import CarND.thresholding as cnthresh
import CarND.calibration as cncalib
import CarND.perstransform as cntransform


# Camera calibration
objpoints, imgpoints, img_size = cncalib.getPointsInfo('camera_cal/calibration*.jpg', patternSize=(9,6))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Undistort chess board images
images = glob.glob('camera_cal/calibration*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = './undist_images/' + fname.split('/')[-1].split('.')[0] + '.jpg'
    cv2.imwrite(write_name, dst)
    print(write_name)

# Apply a distortion correction to raw images
images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = './undist_images/' + fname.split('/')[-1].split('.')[0] + '.jpg'
    cv2.imwrite(write_name, dst)
    print(write_name)

# Use color transforms, gradients, etc., to create a thresholded binary image
# Read in an image
images = glob.glob('test_images/*.jpg')
ksize = 15 # Sobel kernel size - Choose a larger odd number to smooth gradient measurements

for idx, fname in enumerate(images):
    img = mpimg.imread(fname)

    # Apply each of the thresholding functions
    gradx = cnthresh.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = cnthresh.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = cnthresh.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = cnthresh.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    color_binary = cnthresh.color_threshold(img, thresh=(170, 255))

    combined = np.zeros_like(dir_binary, dtype='uint8')
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Combined Thresholded Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

    write_name = './undist_images/' + 'thresh_' + fname.split('/')[-1].split('.')[0] + '.jpg'
    cv2.imwrite(write_name, combined)



# Get perspective transform
img = mpimg.imread('undist_images/thresh_straight_lines1.jpg')
warped_im = cntransform.warp(img)

# Visualize undistortion
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
histogram = np.sum(warped_im[warped_im.shape[0]//2:,:], axis=0)
plt.plot(histogram)
plt.show()

# Create an output image to draw on and visualize the result
out_img = np.dstack((warped_im, warped_im, warped_im))
# plt.imshow(out_img)
# plt.show()

# Fine the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lanes
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9

# Set height of windows
window_height = np.int(warped_im.shape[0]//nwindows)

# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_im.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base

# Set the width of the windows +/- margin
margin = 100

# Set minimum number of pixels found to recenter window
minpix = 50

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped_im.shape[0] - (window+1)*window_height
    win_y_high = warped_im.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
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
ploty = np.linspace(0, warped_im.shape[0]-1, warped_im.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()




# Warp the detected lane boundaries back onto the original image


# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position






print('completed')
