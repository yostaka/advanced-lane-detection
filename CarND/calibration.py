import numpy as np
import cv2
import glob


def getPointsInfo(img_files, patternSize=(8, 6)):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(img_files)
    img_size = (0, 0)

    print("getPointsInfo patternSize: ", str(patternSize[0]), ",",  str(patternSize[1]))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        print("Extract chessboard corners from", fname, " ", end="")

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, patternSize, corners, ret)
            write_name = './output_images/image_output/chessboard_corners/' + fname.split('/')[-1].split('.')[0] + '.jpg'
            cv2.imwrite(write_name, img)
            print(": Corners found")
        else:
            print(": Corners not found")

    return objpoints, imgpoints, img_size


# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


