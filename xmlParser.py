import xml.dom.minidom as xdm
import cv2
import os

xmlFolderPath = "C:\\Users\\carel\\Desktop\\datamark\\FILE0007\\FILE0007_xml"
for root, dirs, files in os.walk(xmlFolderPath):
    for xmlfile in files:
        xmlPath = os.path.join(root, xmlfile)
        dom = xdm.parse(xmlPath)
        rootXML = dom.documentElement
        imagePath = rootXML.getElementsByTagName('path')[0].firstChild.data
        imageName = imagePath.split('\\')[-1]
        allObject = rootXML.getElementsByTagName('object')
        i = 0
        for myObject in allObject:
            i = i + 1
            bndBox = myObject.getElementsByTagName('bndbox')[0]
            impath = "C:\\Users\\carel\\Desktop\\datamark\\FILE0007\\Images\\" + imageName
            img = cv2.imread(impath, 3)
            xmin = int(bndBox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndBox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndBox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndBox.getElementsByTagName('ymax')[0].firstChild.data)
            img2 = img[ymin:ymax, xmin:xmax, :]
            cv2.imwrite("C:\\Users\\carel\\Desktop\\datamark\\FILE0007\\trainData\\" + str(i) + 'of' + imageName, img2)
            print 'success once'

