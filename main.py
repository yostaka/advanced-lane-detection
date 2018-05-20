import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import CarND.thresholding as cnthresh
import CarND.calibration as cncalib
import CarND.perstransform as cntransform
import CarND.lanedetection as cnld


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
out_img = np.dstack((img, img, img)) * 255
# img = mpimg.imread('undist_images/thresh_test2.jpg')

area_img, lane_img = cnld.getLaneMaskImage(img)
out_img = cv2.addWeighted(out_img, 1, lane_img, 1, 0)
out_img = cv2.addWeighted(out_img, 1, area_img, 0.3, 0)

# Warp the detected lane boundaries back onto the original image
# unwarped_im = cntransform.unwarp(result)
plt.imshow(out_img)
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()


# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position






print('completed')
