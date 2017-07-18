import cv2
import numpy as np
from imutils import paths
import imageop as im
TRAIN = True
CROP = False
PosSamNO = 2400    #正样本个数
NegSamNO = 12000
HardExampleNO = 0

hog = cv2.HOGDescriptor()
# print type(cv2.HOGDescriptor_getDefaultPeopleDetector())
# iter = 0
# flag = True
# for iter in range(len(cv2.HOGDescriptor_getDefaultPeopleDetector())):
#     print cv2.HOGDescriptor_getDefaultPeopleDetector()[iter]
#
# print len(cv2.HOGDescriptor_getDefaultPeopleDetector())
myDetector = np.array([[]])
descriptor = 0;
num = 0
if TRAIN:
    imagePath = "D:\\FFOutput\\Thefirst"  # input your video path
    for imagePath in paths.list_images(imagePath):
        print ("Handling image %s ." %(imagePath))
        image = cv2.imread(imagePath)

        if CROP:
            image = im.crop((100, 100, 200, 200))# test
        descriptor = hog.compute(image, (8, 8))
        descriptorDim = len(descriptor)
       #sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, descriptorDim, CV_32FC1);
