# Project Proposal

ENGR3590: *A Computational Introduction to Robotics*

Project II : Robot Learning

## Team Members

Yoonyoung Cho

## Project Definition

In the current form, the project seeks to utilize deep sensor fusion with onboard sensors (lidar/camera/wheel encoder) and commanded velocity to odometry estimates of self-motion (as compared to ground-truth localization). The problem explores the domain of enhancing control signals with richer estimation of self-motion, thereby reducing potential errors that may be induced further down the robot localization and navigation pipelines.

## Potential References

There are plenty of resources out there that explore the concept of deep visual-inertial odometry computation such as [VINet](https://arxiv.org/pdf/1701.08376.pdf), which may be useful (except the project seeks to integrate a larger array of sensors for more robust localization).

One possible extension that would introduce more value to the project would be to incorporate architectures such as [RegNet](https://arxiv.org/pdf/1707.03167.pdf) for automatic multisensor calibration with minimal prior information.

## Development Timeline

The current MVP(Minimum Viable Product) consists of a predictor that takes `cmd_vel` and additional sensor data from the LIDAR or the camera to produce rectified local odometry closer to the ground truth than the raw baseline (raw wheel encoder integration). Temporal information will be processed by stacking the inputs after some initial processing, rather than constructing and processing a frame-by-frame input at each occasion.

For a sophisticated version, the network will incorporate recurrent components to robustly encode state information that persist across temporal gaps.

A possible direction in scaling the model includes recovery from simulated sensor faults (absence of data for a duration of time) and simulated motor faults (*wrong* wheel encoder data for a duration of time).

## Data collection

In order to ensure a wide range of input data, the `cmd_vel` input will be supplied with a mouse-based teleoperation node that acts in the continuous domain without too much efforts. Data will be sampled and possibly aggregated at regular intervals to reduce chances of collecting redundant data.

Given a relatively unsophisticated problem definition, ...

## Algorithm

This was mentioned several times in the previous sections, but I would most likely start with a convolutional neural network operating on a time- and sensor-stacked input. If given enough time and desire for better performance, I may try to migrate the model to a recurrent model which may reduce the amount of computation and the number of heuristics introduced for consideration of temporal states. If the model works fabulously and I have an eternity of time (very unlikely), I may try to work with an end-to-end Reinforcement Learning model that seeks to navigate to a goal position with sparse success signals.

## Baseline algorithm

The *raw* baseline would incorporate simple wheel encoder integration without taking other sensors into account. I can apply a SLAM-based approach as the baseline classical algorithm to provide a reasonable expectation for what the network should be able to achieve - depending on the performance, this result may be fed back into the pipeline as the *label* for the training data. I'll most likely use packages such as `hector_mapping` or `gmapping` for this.