import xml.dom.minidom as xdm
import cv2
import os


batches = ["first", "third"]
for batch in batches:
    xmlFolderPath = "D:\\detectProject\\datamark\\xml\\" + batch
    #xmlFolderPath = "D:\\detectProject\\test\\"
    for root, dirs, files in os.walk(xmlFolderPath):
        for xmlFile in files:
            xmlPath = os.path.join(root, xmlFile)
            print xmlPath
            dom = xdm.parse(xmlPath)
            rootXML = dom.documentElement
            imageNameArray = str(xmlFile).split('MOV')
            allObject = rootXML.getElementsByTagName('object')
            imgFinal = "image" + imageNameArray[0] + imageNameArray[1].split('.xml')[0] + ".jpg"
            imgPath = "D:\\detectProject\\dataframe\\" + batch + "\\" + imgFinal
            img = cv2.imread(imgPath, 3)
            i = 0
            for myObject in allObject:
                i = i + 1
                type = myObject.getElementsByTagName('name')[0].firstChild.data
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

                if y1 > 0 and y2 < 720 and x1 > 0 and x2 <= 1280:
                    img2 = img[y1:y2, x1:x2, :]
                    cv2.imwrite("D:\\detectProject\\traindata\\" + type + "\\" + size + str(i) + batch + imgFinal, img2)

                bgCount = 0
                xc1 = x1
                xc2 = x2
                while xc1-200 > 0:
                    xc1 = xc1-200
                    xc2 = xc2-200
                    img3 = img[y1:y2, xc1:xc2, :]
                    bgCount += 1
                    cv2.imwrite("D:\\detectProject\\negativedata\\" + size + "bg" + str(bgCount) + batch + imgFinal, img3)

                xc1 = x1
                xc2 = x2
                while xc2+200 < 1280:
                    xc1 = xc1+200
                    xc2 = xc2+200
                    img3 = img[y1:y2, xc1:xc2, :]
                    bgCount += 1
                    cv2.imwrite("D:\\detectProject\\negativedata\\" + size + "bg" + str(bgCount) + batch + imgFinal, img3)
                print 'success once' + str(xmlFile)