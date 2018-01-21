# objectDetection
Object Detection using Single Shot Detectors and MobileNet.
The following program mimics a popular object detection project. 
In this program, the use of the Single Shot Detector (SSD) Algorithm and Mobile Net architecture is used.
Usually, when working with object detection 3 algorithms come to mind: SSD, Faster R-CNN, and You Only Look Once (YOLO)
This program focuses on SSD; Future projects will focus on Faster R-CNN and YOLO. 

MobileNet was used over other architectures (like ResNet) as it is not as computation heavy as the other archietctures. Although it is important to note that it is not as accurate, but is definetely more efficent. It was made by Google to run deep learning algorithms on mobiles. 

Here are the research papers of SSD and MobileNet:
SSD: https://arxiv.org/pdf/1512.02325.pdf
MobileNet: https://arxiv.org/pdf/1704.04861.pdf

###############################################
To run the program:
1. Go to directory where ObjectDetect.py exists.
2. On terminal, type:
      python ObjectDetect.py --prototxt MobileNetSSD_deploy.prototxt.txt --preTrainedModel MobileNetSSD_deploy.caffemodel --         imageOfInterest images/choiceOfImage
