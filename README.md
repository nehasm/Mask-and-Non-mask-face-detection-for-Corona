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

1. Create annotations using tool labelimg which create annotations in pascalvoc format.
2. Convert annotations into fiyzr format by 
   a. create a zip file containing training dataset images and annotations with the same filename 
   b. Upload zip file in Google Drive, get Drive file id, and substitute the DATASET_DRIVEID value.
   c. Run cell that iterates over the xml files and creates annotations.csv file
3. Model training:
    Fizyr offers various parameters, described in [Github](https://github.com/fizyr/keras-        retinanet/blob/c841da27f540084d27e971b6d00c178ff005d344/keras_retinanet/bin/train.py#L358), to run and optimize this step.

    It’s a good option to start from a pretrained model instead of training a model from scratch. Fizyr released a model based on     ResNet50 architecture, pretrained on Coco dataset.





