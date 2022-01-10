# Introduction

Facebook AI Research (FAIR) introduced the Detectron2 package, which produces excellent performance on object detection and segmentation issues. Detectron2 is written in PyTorch and is based on the maskrcnn benchmark. It requires CUDA because the computing costs are rather high.
Our project begins with the creation of a Machine Learning model that can distinguish between five different vehicle classes: automobile, bus, motorcycle, ambulance, and truck. We used an open-source dataset accessible on the Roboflow website to train a Faster R-CNN model on our dataset over a baseline model given by detectron2 and trained by experimenting with various sets of hyperparameters to produce a more accurate model. There are approximately 830 photos in our dataset, which are divided into test, validation, and train subsets. The photos have a resolution of 1024x751 pixels, and the car appears to be overrepresented in the sample. COCO-Dataset format is required by Detectron2, which is essentially a JSON file holding metadata about the photos, such as labels, classes, dimensions, and so on.
The experiment's second portion shows how multiple detectron2 pre-trained models may be used to conduct direct inference on new data for tasks including image segmentation, key-point identification, LVIS segmentation, Panoptic segmentation.

# Overview of Faster R-CNN model

Before we can train our data, we need to import a baseline model from Detectron2 and set up the model's configurations. For our training, we use a Faster R-CNN model. A group of Microsoft researchers developed the Faster R-CNN model, which is a deep convolutional network used for object detection that appears to users as a single end-to-end unified network. Faster R-CNN constructs a region-proposal network that can generate region proposals, which are then input into the detection model (Fast R-CNN) for object inspection.
The general architecture of Faster R-CNN is shown below, and it consists of 2 modules:

![image](https://user-images.githubusercontent.com/33214665/148707396-63b36b28-3a16-4231-a80a-e99e53fdc9d5.png)

1. RPN: For generating region proposals, the Fast R-CNN detection module is guided to where to seek for items in the image by using the idea of attention in a neural network.
2. R-CNN Fast: For finding items in the suggested regions. The ROI pooling layer extracts a fixed-length feature vector from each region once the region proposal is formed for all region proposals in the pictures. Fast R-CNN is used to classify the retrieved features, and the class scores of the discovered objects, as well as their bounding boxes, are returned.

# Experimental Setup

We use Google Colab to do our coding and train our model. The following set of libraries, with their following specific versions, are required to support Detectron2 [5]:
1. Torch v1.5.0
2. TorchVision v0.5
3. Pyyaml v5.1
4. Cocoapi
5. Pycocotools-2.0
6. Numpy
7. Matplotlib and OpenCV libraries for visualizing

# Dataset

The photos must be in 'COCO format' for Detectron2 to work. The "COCO format" is a JSON structure that contains information on how labels and metadata for an image dataset are saved. We get our data from the open-source Roboflow public datasets [8], which allow users to add their tailored data. We use the following commands to download and extract the data directly in Colab:
!curl -L "https://public.roboflow.com/ds/20BnsvX8UE?key=HbdJ3kaKcQ" > roboflow.zip


# Results

![image](https://user-images.githubusercontent.com/33214665/148707458-228e9ccb-ea54-47cd-9ea9-99028b6ff4c0.png)

![image](https://user-images.githubusercontent.com/33214665/148707460-68c95b42-2fd7-405b-9bda-4129f39b0167.png)
