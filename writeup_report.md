# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Additionally model.py imports the following file:
* nvidia.py containing an implementation of the nvidia NN model
* generator.py containing the generator function for training the network
* balance_samples.py containg a function for balancing samples

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the nvidia model presented in the lessons and it seems to be adequate for reaching the goals of the project.

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting:
* I augmented the data set by balancing it and flipping the images
* I properly reduced the number of epochs to the minimum for getting good results

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I corrected the steering angle for the last two by adding/subtracting a correction factor.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First of all i balanced the data set, obtaining the same number of samples for each steering value.

In order to gauge how well the model was working, I shuffled the balanced samples and split a subset of them into a training and validation set. I had to use a subset in order to optimize the time but better results could be obtained by using the entire balanced data set.

In order to optimize the efficiency of the algorithm I created a generator for the training and validation set. It is located in generator.py file and it uses all the three images for each sample and furthermore flip them in order to augment the data set.

I compiled the nvidia model (located in nvidia.py) with an Adam optimized and trained it by using:
```sh
model.fit_generator
```

The final step was to run the simulator to see how well the car was driving around track one. The firt times there were two curves that my car was not able to front, balancing the data set has been the key for achieving better results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

As already reported above I choosed the nvidia model for this project.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data set already present in the workspace without using the simulator for collecting more data. Here is an example of the center, left and right camera images in a line of the provided data set:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would improve the performances of my model and teach it to front left and right curves in the same way.

![alt text][image5]
![alt text][image6]

After the balancing process, I had about 540000 data points. I decided to shuffle them and use just a subset to lower the training time, so i passed just 50000 data points to the generators, splitted in 40000 for training and 10000 for validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph in lossperepoch.png. I used an adam optimizer so that manually training the learning rate wasn't necessary.
