import xml.dom.minidom as xdm
import cv2
import os

xmlFolderPath = "D:\\detectProject\\datamark\\xml"
for root, dirs, files in os.walk(xmlFolderPath):
    for xmlFile in files:
        xmlPath = os.path.join(root, xmlFile)
        dom = xdm.parse(xmlPath)
        rootXML = dom.documentElement
        imageName = str(xmlFile).split('MOV')
        allObject = rootXML.getElementsByTagName('object')
        imageNameFinal = 'image' + imageName[0] + imageName[1].split('.xml')[0] + '.jpg'
        imgPath = "D:\\detectProject\\dataframe\\" + imageNameFinal
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
            cv2.imwrite("D:\\detectProject\\traindata\\" + str(i) + 'of' + size + imageNameFinal, img2)
            print 'success once' + str(xmlFile)

