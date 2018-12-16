# Robot Learning Project

Yoonyoung (Jamie) Cho | CompRobo FA2018 @ Olin College

## Guidelines

How did you solve the problem?  You should touch on the data you used, the algorithms you applied, etc.  If you implemented your own version of an algorithm as a step in your learning process, make sure to explain what you did for that part of the project as well.
Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.
What if any challenges did you face along the way?
What would you do to improve your project if you had more time?
Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

## Introduction

In the *Robot Learning* Project, my objective was to implement *monocular visual odometry*.

In traditional computer vision approaches, monocular visual odometry is considered a difficult problem: among the number of challenges, the lack of scale information renders the problem inherently under-constrained. Many authors have tackled this problem through approaches such as parallax-based scale estimation, ground-plane estimation with fixed camera heights, and more. Often, the problem will be worked around through employing additional sensors such as an IMU for visual-inertial odometry, or adding another camera to achieve stereo vision.

With the recent advances in deep learning, approaches such as [DeepVo]() has demonstrated compelling performance on monocular visual odometry; in this exploration, I sought to evaluate both modern and classical approaches to familiarize myself with the algorithms, as well as the challenges along the way.

## Data

Throughout this project, multiple different datasets have been utilized at different stages of development. Here, each of them will be addressed and credited appropriately.

- **[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)** : The KITTI Visual Odometry dataset consists of 22 sequences of stereo-camera data, driving around in urban environments.
- **[ILSVRC Dataset](http://image-net.org/challenges/LSVRC/2015/index#vid)** This is a custom-generated large scale optical flow dataset based on the [ILSVRC](http://image-net.org/challenges/LSVRC/2015/index#vid) detection-from-video dataset, processed with a pre-trained [FlowNet](https://arxiv.org/abs/1504.06852) network.
- **[Academic Center Dataset]()** This is a custom-generated small-scale visual odometry dataset in the Olin College Academic Center with low-fidelity *"ground truth"* data from wheel encoder odometry, with images, timestamps and (for a subset) scans associated with each frame.
- **[Gazebo Dataset]()** This is a dataset generated through renders in the [Gazebo](http://gazebosim.org/)-simulated environment of the Olin College Oval, with randomly added objects for readily trackable visual features.
- **[Flying Chairs Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)** : This is a somewhat "classical" dataset employed during the training of the original FlowNet architecture; the dataset consists of ground-truth optical flow associated with an artificially generated displacements of chairs on random background images extracted from Flickr.

The *classical* variant of the algorithm did not require training, and most of the validation was performed on the *Academic Center* dataset and *Gazebo* dataset.

The deep-learning approach of the algorithm was first trained on the optical flow images from a mixture of the *Flying Chairs* dataset and the *ILSVRC* dataset; afterwards, the weights from the flownet portion of the network was transferred to train the rest of the inference network for estimating visual odometry on the *KITTI* dataset. This network was then fine-tuned on the *Academic Center* dataset for scale adjustments.

Both the AC Dataset and the Gazebo Dataset have been generated with [data\_collector.py](scripts/data_collector.py), made available under ```scripts```.

### Data Augmentation

In the case of deep-learning based approaches, it is typical for classification or detection datasets to benefit from a large-scale dataset of millions of annotated images with high-quality labels. Such a dataset was not readily available, although a [compiled summary](https://github.com/youngguncho/awesome-slam-datasets) is available for the curious reader.

As the quantity of data was insufficient for either the optical-flow or visual-odometry tasks, it was imperative that existing data had to be augmented.

While such techniques must be used at caution in order to prevent the network from overfitting to augmented-style inputs, at validation the network did not exhibit significant overfitting behavior (at the final stage, at least).

In this particular case, the augmentation techniques were constrained by the following criteria:

- Augmentation for the images must be consistent across time, but may be different across batches.
- Augmentation for Optical Flow must account for any resizing artifacts on the source image on the flow as well, such that the flow-vector magnitude would be preserved.
- Augmentation for Visual Odometry could not employ any affine transforms on the image, such as cropping, would impact the relevant view-point of the camera.

Here are two samples of an image from the datasets, augmented for training the respective networks:

|![opt_aug](figs/opt_aug.png)|![opt_aug](figs/vo_aug.png)|
|:---:|:---:|
|(a) Optical Flow Augmentation | (b) Visual Odometry Augmentation|

Note that in (a) the scale of the flow vector remains consistent through the cropping operation, and in (b) the spatial structure is preserved for both the reconstructed path and the image itself, so that the original information could be recovered despite the augmentation.

## Algorithms

### Classical

The *classical* algorithm was mostly an exploration with many of the available [OpenCV](https://opencv.org/) libraries and online tutorials, although none of the guides really solved the problem to the level of rigor that I had been hoping for. While many of the high-performing papers were reliant on global or semi-global optimization techniques including bundle adjustment, I didn't delve into the graph optimization portion too heavily on this project; I had planned on implementing the bulk of the critical pieces, and including SBA would have been a significant undertaking for the time being.

On a high level, the algorithm operates on a sparse set of feature points computed with detectors such as [ORB](TODO) and [Shi-Tomasi Detector](TODO), whose keypoints are then converted into feature-vectors with the descriptors such as [BRISK](). These features points are tracked and matched across frames, whose correspondences allow for triangulation to jointly estimate the point's coordinate and the camera's relative poses across the frames.

The persistent tracking of landmarks allow for scale tranfer from an old to new set of landmarks, whereby the initial scale information is preserved across time. However, a frame-by-frame estimate of scale may be desirable, considering that significant drift is introduced for each frame in the PnP optimization process.

#### Results

|![cla_demo](figs/vo/classical_demo.gif)|
|:---:|
| Fig.x. Classical Optical Flow Demo |


#### Design Decisions

- UKF Parameter Tuning
- Opt Flow filtering / detecting failures
- Flann Tuning / Parameters
- Corner Sub-Pixel Alignment
- Landmark PnP Coordinate Frame - Oldest vs Newest
- Re-initialization thresholds
- Landmark incremental updates and smoothing
- Frame-by-frame landmark registration
- Using Extrinsic Guess for RANSAC from UKF
- Choice of PnP Algorithm

TODO : fill

#### Challenges and Next Steps

- Feature Points Uniform Distribution
- Groundplane scale estimation?
- Parallax scale estimation
- Graph Optimization / Bundle Adjustment Steps

### Deep-Learning

The *deep-learning* algorithm was heavily inspired by the [DeepVo](https://arxiv.org/abs/1709.08429) paper, although no "working" implementation was available online as far as I was aware of. Many of the network hyper-parameters and the training configurations were modified to suit for the particular use-case: the images were heavily downsized, and the network made heavy use of Separable Convolution and Batch Normalization where appropriate, unlike the original architecture.

In particular, the network architecture consists of an initial CNN(Convolution Neural Network) portion to produce condensed feature-vector representations, which are then fed through the RNN(Recurrent Neural Network) with stacked LSTM(Long Short-Term Memory) Cells to produce the final representation of 512 floating-point numbers, which are then connected with a series of zero-bias fully connected layers to produce the frame-by-frame translation and rotation in SE2 (for simplicity).

As for the optical-flow network, the net started with the identical CNN layer whose output was processed through cross-correlation across time-steps. This "merged" feature was then processed with the successive part of the network that formed a somewhat hourglass-like construction with [up-convolution]() layers to build up a pyramid of dense optical flow vector fields at each scale, which were all concatenated and convolved iteratively to produce the final full-sized optical flow output. 

#### Results

|![opt_aug](figs/train_plot.png)|![opt_aug](figs/sample_output.png)|
|:---:|:---:|
|(a) Optical Flow Training Plot | (b) Optical Flow Sample Result|


|![opt_aug](figs/vo/train_v2.png)|![opt_aug](figs/vo/sample_output_v2.png)|
|:---:|:---:|
|(a) Visual Odometry Training Plot | (b) Visual Odometry Sample Result|

#### Design Decisions

- SE2
- Downsize
- Separable Convolution
- Batch Norm
- L2 Regularization
- Up-Convolution
- Dynamic weight decay for multi-objective networks
- Cyclic Learning Rate Decay
- Learning Initial "Ramp"
- Dropout Regularization
- VO SE2 Weight Scaling
- Freezing vs fine-tuning pre-trained weights
- Input Queue

#### Challenges and Next Steps

TODO : fill

## Conclusion

TODO : fill