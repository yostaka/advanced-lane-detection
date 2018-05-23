**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img-cal1]: ./camera_cal/calibration2.jpg "Original chessboard"
[img-cal2]: ./output_images/image_output/chessboard_corners/calibration2.jpg "Chessboard with detected corners"
[img-cal3]: ./output_images/image_output/chessboard/calibration2.jpg "Undistored chessboard"

[img-pipeline0]: ./test_images/test1.jpg "Original image"
[img-pipeline1]: ./output_images/image_output/pipeline_images/test1_1_binary.jpg "Binary thresholded image"
[img-pipeline2]: ./output_images/image_output/pipeline_images/test1_2_warped.jpg "Warped image"
[img-pipeline3]: ./output_images/image_output/pipeline_images/test1_3_histogram.jpg "Histogram"
[img-pipeline4]: ./output_images/image_output/pipeline_images/test1_4_laneplots.jpg "Lane plots"
[img-pipeline5]: ./output_images/image_output/pipeline_images/test1_5_lanelines.jpg "Lane lines"
[img-pipeline6]: ./output_images/image_output/pipeline_images/test1_6_lanearea.jpg "Lane area"
[img-pipeline7]: ./output_images/image_output/pipeline_images/test1_7_overlayed.jpg "Overlayed image"

[img-parallel-warped]: ./output_images/image_output/pipeline_images/straight_lines1_2_warped.jpg "Warped paralled lanes"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Code Structure

Here's the code structure for this project:

```
.
+--- camera_cal/    # Chessboard images
|  +--- calibration*.jpg
|
+--- CarND/         # Functions built for lane detection
|  +--- calibration.py
|  +--- lanedetection.py
|  +--- perstransform.py
|  +--- thresholding.py
|
+--- output_images/
|  +--- image_output/   # Image outputs
|  |  +--- chessboard/          # Undistorted chessboard images
|  |  +--- chessboard_corners/  # Chessboard images with markers on corners
|  |  +--- pipeline_images/     # Pipeline images ing process
|  |  +--- results/             # Detected car lane + carveture and car position info on original images
|  |
|  +--- video_output/   # Video outputs
|
+--- test_images/   # Original images
|  +--- *.jpg
|
+--- main.py    # Main code - entry point for the car lane detection
```

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for computing camera matrix, distortion coefficients and generating corrected calibration images is contained in `main.py` and `./CarND/calibration.py`.

The line 13 in `main.py` calls `cncalib.getPointsInfo()` function defined in `/CarND/calibration.py`, which generates object points and image points using chessboard images. Then `cv2.calibrateCamera()` function at line 4 in `main.py` computes the camera matrix `mtx` and distortion coefficient `dist`. I applied these parameters to the test images using the `cv2.undistort()` function and obtained this result:

Original chessboard
![alt text][img-cal1]

Chessboard with detected corners
![alt text][img-cal2]

Undistorted chessboard
![alt text][img-cal3]

Here's the detailed processing in the `cncalib.getPointsInfo()` function:

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][img-pipeline0]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 45 through 54 in `lanedetection.py`).  Here's an example of my output for this step. At last step, I masked image area irrelevant to lane detection to reduce noise to lane detection algorithm. As a result, Masked Combined Image is used as input to detect lane lines.

![alt text][img-pipeline1]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 5 through 27 in the file `perstransform.py` (CarND/perstransform.py).  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[761, 499],
     [1034, 673],
     [277, 673],
     [528, 499]])

dst = np.float32(
    [[1034, 499],
     [1034, 673],
     [277, 673],
     [277, 499]])
```

First, I verified using straight line images to check if the warped image have parallel lane lines.

![alt_text][img-parallel-warped]

Then, I applied warping process to other images using same src and dst parameters. Follwing figure shows one of the images with curve applying the warping process.

![alt text][img-pipeline2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for lane detection are in a function called `getLaneMaskImage()` and lane detection steps appears in line 108 through line 279 in the file `lanedetection.py` (CarND/lanedetection.py). 

I describes steps for lane detection as follows.

1. Take histogram counting number of pixels in same x coordinate. I used the 2 peaky x coordinates as starting points for left lane line and right lane line respectively. To be more precise, left lane detection starts with x coordinate having highest number of counts on left half area, and right lane detection starts with x coordinate having highest number of counts on right half area.

![alt text][img-pipeline3]

2. I used windowing method to determine which pixels are relevant to each lane lines, and it allows to follow curved lanes. Then, I fit my lane lines with a 2nd order polynominal.
![alt text][img-pipeline4]
![alt text][img-pipeline5]

3. I filled area in green between detected left lane line and right lane line.
![alt text][img-pipeline6]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 285 through 297 in my code in `lanedetection.py`

To culculate the radius of curvature of the lane, I converted the measuring units from pixel to meters. I used 60/720[m/px] as vertical direction (y-coordinate) and 3.7/720[m/px] as horizontal direction (x-coordinate). For horizontal direction, I used US standard lane width (3.7 meters). For vertical direction, I adjusted the number so the radius of curvature result in roughly right radius.

Then, I used equation to calculate radius of curvature. For more details, https://www.intmath.com/applications-differentiation/8-radius-curvature.php is good reference to understand the theory.


To culculate the position of the vehicle, I suppose that the camera is installed in the middle of the car, meaning that the center of the image corresponds to the center of the vehicle.

First, I counted the number of pixels between most left of the image and detected left lane line at the bottom, and the number of pixels between most right of the image and detected right lane line at the bottom. I subtracted those two values to derive how many pixels the car is positioned relative to the center of the image. Then, I converted the unit of the distance from pixel to meter using 3.7/720[m/px]. The code for this process are lines 295 through line 297 in `lanedetection.py`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 266 through 282 in my code in `lanedetection.py` in the function `getLaneMaskImage()`.  Here is an example of my result on a test image:

![alt text]![alt text][img-pipeline7]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/video_output/lane_detection8.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For the curving road, the lane detection I implemented so far calculate lines as excessive steep curve than the one acutually is. This is caused by very few information on dashed car lane markers, which tend to make detected curve excessive. I could use smoothing algorithm among adjacent frames, which can mitigate such outlier frames.
