import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import CarND.calibration as cncalib
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

for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    combined = cnld.getThresholdedBinaryImage(img, show_img=True)

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
