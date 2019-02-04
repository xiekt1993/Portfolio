# Traffic Detection in Real-time Streaming Footage
## Introduction
The goal of this project is to be able to draw bounding boxes for all cars in dashboard camera footage. Application for such technology, include autonomous vehicles where localization of vehicles is needed to navigate traffic and smart traffic cameras that can monitor the flow of traffic on a busy highway. The real-time monitor application provides a powerful tool for transportation agency to record and analyze traffic data in another way.
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/Screen-Shot11.jpg" width="750"/>
</p>

## Implementation
A deep convolution neural network is trained with a larger dataset and more convolution layers in order to improve the test performance. There are many famous deep learning object detection algorithms in the research and practical areas, including the very popular ones Faster-RCNN (Ren et al., 2017) and YOLO model (Redmon et al., 2016). They all seek to solve a regression (bounding boxes) and a classification task at the same time. The main difference is how they achieve these two goals in their network architectures. In comparison, the YOLO model has higher training speed but lower prediction scores. Considering the nature of our car detection task, the speed as well as the real-time performance matter more than the prediction since we only focus on one class. Therefore, we chose YOLO model as our deep net.
- Dataset
Here we used PASCAL VOC 2012 dataset as our training dataset, which is a well-known but simple dataset for object detection and segmentation. The dataset contains about 20 categories and 10,000 images for training and validation, each annotated with bounding box and object category. And Figure 12 demonstrate some examples from the dataset. We also implemented data augmentation techniques to avoid overfitting as described in next section.
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/dataset.jpg" width="750"/>
</p>
- Data Augmentation
- Model
The model we implemented can be referenced in the paper YOLO9000: Better, Faster, Stronger (Redmon et al., 2016). The author proposed a Darknet-19 model, which has 19 convolution layers and 5 max-pooling layers.
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/model.jpg" width="200"/>
</p>

## Results
Due the capacity of our computation, we are only able to train a few epochs during the training. The training loss decreases very rapid in just few epochs, but we are also facing a very high testing overfitting error on the classification. And then we used a video clip captured on a vehicle driving on highway as our test case (see below). Since we have trained the model on 20 categories and then predicted on only one class with lower confidence threshold, we are hoping this could mitigate the overfitting effect. As a result, the test video managed to capture some information about the cars as the target object but there are also a number of false-positive predicted boxes. Besides, the model also gets confused when there are overlapping objects. We are hoping that add more training epochs or hyper-parameter fine tuning would help improve the performance. 
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/footage3.jpg" width="200"/>
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/footage1.jpg" width="200"/>
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/footage4.jpg" width="200"/>
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/footage2.jpg" width="200"/>
</p>

## Combined with ArcGIS for real-time monitor
After conducting the object detection on traffic videos, we can export the information and create a leaflet dashboard to record and visulize the information.
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/Screen-Shot1_prediction.jpg" width="700"/>
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Car_Detection_in_Traffic_Videos/Screen-Shot2_prediction.jpg" width="700"/>
</p>

## Future Improvement
- To improve the test accuracy, the model would make use of the temporal structure of the dashboard camera footage, such as adding RNN architecture.
- On the other hand, the model only trained on a relative small dataset. Future improvements can be made on training on a larger one.
- When combining with ArcGIS, more data could be added for heat map and real-time data dashboards.

## Reference
- Ren, Shaoqing, et al. "Faster R-CNN: towards real-time object detection with region proposal networks." IEEE Transactions on Pattern Analysis & Machine Intelligence 6 (2017): 1137-1149.
- Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." arXiv preprint (2017).
