import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

import CarND.calibration as cncalib
import CarND.lanedetection as cnld


# Configurations
generateVideo = False
video_input = 'project_video.mp4'
video_output = 'output_images/video_output/lane_detection8.mp4'


# Camera calibration
objpoints, imgpoints, img_size = cncalib.getPointsInfo('camera_cal/calibration*.jpg', patternSize=(9,6))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Undistort chess board images and save them to output_images folder
images = glob.glob('camera_cal/calibration*.jpg')
print('Generating undistorted chessboard images...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = './output_images/image_output/chessboard/' + fname.split('/')[-1].split('.')[0] + '.jpg'
    cv2.imwrite(write_name, dst)


# Generate lane detection result images for all jpg files under test_images folder
# and save the result image and step-by-step processing results in the pipeline
images = glob.glob('test_images/*.jpg')

for idx, fname in enumerate(images):
    print("Processing lane detection on", fname)
    img = mpimg.imread(fname)
    out_img = cnld.getOverlayedImg(img, mtx, dist, show_img=True, name=fname.split('/')[-1].split('.')[-2], save_folder='./output_images/image_output/pipeline_images/')

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original: ' + fname.split('/')[-1], fontsize=50)
    ax2.imshow(out_img)
    ax2.set_title('Overlayed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    write_name = './output_images/image_output/results/' + fname.split('/')[-1]
    plt.savefig(write_name)

print("")


# Build video clip with lane detection

def process_image(image):
    return cnld.getOverlayedImg(image, mtx, dist)


if generateVideo is True:
    clip1 = VideoFileClip(video_input)
    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(video_output, audio=False)

print('Completed all processes')
