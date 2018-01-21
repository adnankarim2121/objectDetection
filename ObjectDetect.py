#Author: Adnan Karim
#This is using MobileNets + SSD. Link will be in the 'read me'.
#Begin by importing the necessary packages for the project
import numpy as np
import argparse 
import cv2 #using openCV 3.4.0

#Making use of the argument parser library
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--imageOfInterest", required = True, help = "Put your path to the image file here")
argparser.add_argument("-p", "--prototxt", required = True, help="path to Caffe 'deploy' prototxt file" )
argparser.add_argument("-pm", "--preTrainedModel", required = True, help = "path to Caffe pre-trained model")
argparser.add_argument("-prob", "--probabilty", type = float, default = 0.2, help = "the lowest probabilty needed to filter out weak object detections")

args = vars(argparser.parse_args())

#Next,initialize the list of class labels that our MobileNet SSD will detect.
detectClasses = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
#Now, create a bounding box for each label/classes; they will have distinct colors
colorForClasses = np.random.uniform(0, 255, size = (len(detectClasses), 3))

#loading model with the next two lines
print("Loading model...")
mNet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["preTrainedModel"])

#now, load the input image.
imageOfInterest = cv2.imread(args["imageOfInterest"])
#gran height and width of image
(height, width) = imageOfInterest.shape[:2]
#once image is loaded, construct an input blob for the image by resizing it into 300x300 pixels
#and then normalizing it (normalization is done by the authors of MobileNet SSD, which is linked in the readMe)
imageBlob = cv2.dnn.blobFromImage(cv2.resize(imageOfInterest, (300,300)), 0.007843, (300,300), 127.5)


#since we have the image blob (regions in a digital image where properties differ [brightness, color])
#train the imageBlob with a neural network
mNet.setInput(imageBlob)
objectDetections = mNet.forward()

#once object detections are found, loop over them
for i in np.arange(0, objectDetections.shape[2]):
	#get the probability associated with the prediction
	probabilityOfObjects = objectDetections[0, 0, i, 2]

	#if our detection probabilty is greater than our thresholded probabilty value (20%), then
	#keep them for further evaluation

	if probabilityOfObjects > args["probabilty"]:
		#get the index of the class labels from the object detections
		index = int(objectDetections[0, 0, i, 1])
		#once that is done, compute the (x,y) coordinates for the bounding box, which will bound the 
		#object(s) of interest
		boundingBox = objectDetections[0, 0, i, 3:7] * np.array([width, height, width, height])
		(firstX, firstY, secondX, secondY) = boundingBox.astype("int")

		#display predictions (on console)
		result = "{}: {:.2f}%".format(detectClasses[index], probabilityOfObjects*100)
		print("{}".format(result))
		#display predictions (on the image)
		cv2.rectangle(imageOfInterest, (firstX, firstY), (secondX,secondY), colorForClasses[index], 2)
		#this next calculation is just regarding where to put the prediction in the rectangle displaying the result on the image
		y = firstY - 15 if firstY - 15 > 15 else firstY + 15
		cv2.putText(imageOfInterest, result, (firstX, firstY), cv2.FONT_ITALIC, 0.5, colorForClasses[index],2)

#once all the results have been processed, display the output image with the results
cv2.imshow("Output of network", imageOfInterest)
cv2.waitKey(0)

