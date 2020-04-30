# Mask-and-Non-mask-face-detection
A deep learning model to detect mask and non mask wearing people.

We all are dealing with global pandemic.In such situation,Mask is prove to be the best defender from virus.
Detection of mask will help community to monitor mask and non mask wearing people in real time with high level AI model.
I had made a mask and non mask wearing face detection model.
The model is build over the wonderful keras implementation of retinanet object detection developed by [fiyzr](https://github.com/fizyr/keras-retinanet).


Overview of retinanet

RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a conv feature map over an entire input image and is an off-the-self convolution network. The first subnet performs classification on the backbones output; the second subnet performs convolution bounding box regression.

Backbone: Feature Pyramid network built on top of ResNet50 or ResNet101.

Classification subnet: It predicts the probability of object presence at each spatial position for each of the A anchors and K object classes. Takes a input feature map with C channels from a pyramid level, the subnet applies four 3x3 conv layers, each with C filters amd each followed by ReLU activations. Finally sigmoid activations are attached to the outputs. Focal loss is applied as the loss function.

Box Regression Subnet: Similar to classification net used but the parameters are not shared. Outputs the object location with respect to anchor box if an object exists. smooth_l1_loss with sigma equal to 3 is applied as the loss function to this part of the sub-network.
To know more about model working view [here](https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d).

Mask Detection Model

The Data for mask and non mask wearing faces was not available anywhere.I made the dataset which is now available at kaggle [here].

Steps to develop model:
* Make sure to have image dataset
* The images need to be labeled which can be done by using [labelimg](https://github.com/tzutalin/labelImg).
* Make sure to fulfil the input requirements for the model
* Run the model by changing the path as given in usage.

Usage:





