**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/output_bboxes.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 166 through 183 of the file called `project.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. The testing criteria were the quality of recognition, the accuracy of trained classifier and the speed of calculation. The combination of parameters I used for the creation of project video was this:

```
colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
```

I created a class `FeatureExtractionArgs` for storing the selected values which serves as a data transfer object for communication between the functions. The class `FeatureExtractionArgs` provides two methods for feature extractions:

* `extract_feature_category`: Extracts the features for one category of training/test data
* `extract_features`: Extracts the features for both categories (car/not-car) of training/test data.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using ``LinearSVC`` class. The implementation is located in the method ``train_classifier`` of the class ``FeatureClassification``, lines 129 through 132. The data preparation is the responsibility of ``prepare_features`` of the class ``FeatureExtraction``. I decided not to use color features since the difference between using them and not using them was hardly observable. At the end the HOG features turned out to be far more significant for the vehicle recognition. Also I have encountered spurious failures in ``sklearn.preprocessing.StandardScaler`` while trying to scale the color and gradient feature vectors. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used an adaptation of the ``find_cars`` method from the lessons. I removed the part dealing with color features and added the adapter methods in ``FeatureClassification``:

* ``find_cars``: Invokes the actual implementation after setting up the parameter vector.
* ``find_cars_multiscale``: Performs the multi-scale search using limits and scales determined by exerimentation.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales (1.0, 1.5, 2.0, 3.5) using YUV 3-channel HOG features without spatially binned color and histograms of color in the feature vector, which provided an acceptable result. Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The relevant code is in functions ``add_heat`` and ``apply_threshold`` which were taken from the lessons. In the function ``process_image`` I have recorded a history of boxes identified in 20 consecutive frames and applied the heatmap filter to all of them. I have experimented with different values for thresholds. The stability of the detection has improved.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example of a test image and the corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the original image:

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

I am not very satisfied with the stability of the detection pipeline. I was hoping that heatmap filter applied to the sequence of boxes recognized in previous 20-30 steps would help but I did not succeed to find suitable parameters. If I have had more time I'd probably try to track the position of boxes on a bird's view transformed image using a physics based predication model in which the estimated velocity and trajectory of the vehicles are taken into account.

