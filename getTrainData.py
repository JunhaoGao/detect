import cv2
import os
import xml.dom.minidom as xdm

videoPath = "D:\\detectProject\\data\\first" #input your video path
for root, dirs, files in os.walk(videoPath):
    for videoFile in files:
        videoCapture = cv2.VideoCapture(os.path.join(root, videoFile))
        fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        print os.path.join(root, videoFile)
        success, frame = videoCapture.read()
        #frame = cv2.resize(frame, (480, 320)) #zip the frame
        c = 0
        while success:
            if (c % 5 == 0):
                #frame = cv2.resize(frame, (480, 320)) #zip the frame
                cv2.imwrite('D:/detectProject/dataframe/' + 'image' + str(videoFile).split('.MOV')[0] + str(c) + '.jpg',
                            frame)
            success, frame = videoCapture.read()
            c = c + 1


xmlFolderPath = "D:\\detectProject\\datamark\\xml"
for root, dirs, files in os.walk(xmlFolderPath):
    for xmlFile in files:
        xmlPath = os.path.join(root, xmlFile)
        dom = xdm.parse(xmlPath)
        rootXML = dom.documentElement
        imagePath = rootXML.getElementsByTagName('path')[0].firstChild.data
        imageName = imagePath.split('\\')[-1]
        imageNameArray = str(imageName).split('.MOV')
        allObject = rootXML.getElementsByTagName('object')
        imgPath = "D:\\detectProject\\dataframe\\" + imageNameArray[0] + imageNameArray[1]
        img = cv2.imread(imgPath, 3)
        i = 0
        for myObject in allObject:
            i = i + 1
            bndBox = myObject.getElementsByTagName('bndbox')[0]
            xmin = int(bndBox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndBox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndBox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndBox.getElementsByTagName('ymax')[0].firstChild.data)
            xavg = (xmin + xmax)/2
            yavg = (ymin + ymax)/2
            if (ymax - ymin) <= 100:
                y1 = int(yavg - 50)
                y2 = int(yavg + 50)
                x1 = int(xavg - 25)
                x2 = int(xavg + 25)
                size = 'S'
            elif 100 < (ymax - ymin) <= 200:
                y1 = int(yavg - 100)
                y2 = int(yavg + 100)
                x1 = int(xavg - 50)
                x2 = int(xavg + 50)
                size = 'M'
            elif (ymax - ymin) > 200:
                y1 = int(yavg - 200)
                y2 = int(yavg + 200)
                x1 = int(xavg - 100)
                x2 = int(xavg + 100)
                size = 'L'
            img2 = img[y1:y2, x1:x2, :]
            cv2.imwrite("D:\\detectProject\\traindata\\" + str(i) + 'of' + size + imageName, img2)
            print 'success once' + str(xmlFile)

