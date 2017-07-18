import xml.dom.minidom as xdm
import cv2
import os

xmlPath = "C:\\Users\\GarenGao\\Desktop\\datamark\\FILE0007\\FILE0007_xml"
for root, dirs, files in os.walk(xmlPath):
    for file in files:
        dom = xdm.parse(os.path.join(root, file))
        root = dom.documentElement
        imagePath = root.getElementsByTagName('path')[0].firstChild.data
        imageName = imagePath.split('\\')[-1]
        allObject = root.getElementsByTagName('object')
        i = 0
        for myObject in allObject:
            i = i + 1
            bndBox = myObject.getElementsByTagName('bndbox')[0]
            impath = 'C:\\Users\\GarenGao\\Desktop\\datamark\\FILE0007\\Images\\' + imagePath
        !!!    img = cv2.imread('C:\\Users\\GarenGao\\Desktop\\datamark\\FILE0007\\Images\\' + imagePath, 3)
            xmin = bndBox.getElementsByTagName('xmin')[0].firstChild.data
            ymin = bndBox.getElementsByTagName('ymin')[0].firstChild.data
            xmax = bndBox.getElementsByTagName('xmax')[0].firstChild.data
            ymax = bndBox.getElementsByTagName('ymax')[0].firstChild.data
            img2 = img[xmin:xmax, ymin:ymax, :]
            cv2.imwrite('C:\\Users\\GarenGao\\Desktop\\datamark\\FILE0007\\trainData\\' + str(i) + 'of' + imagePath,
                        img2)


 