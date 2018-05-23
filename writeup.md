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


[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

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

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][img-pipeline1]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][img-pipeline2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][img-pipeline3]
![alt text][img-pipeline4]
![alt text][img-pipeline5]
![alt text][img-pipeline6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text]![alt text][img-pipeline7]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/video_output/lane_detection8.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
