import xml.dom.minidom as xdm
import os


def read_xml(xmlpath):
    xmlimglist = []
    dom = xdm.parse(xmlpath)
    rootxml = dom.documentElement
    objects = rootxml.getElementsByTagName('object')
    i = 0
    for myobject in objects:
        i += 1
        bndbox = myobject.getElementsByTagName('bndbox')[0]
        xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        xmlimglist.append((xmin, ymin, xmax, ymax))
    return xmlimglist


def validate_list(txtimagelist, xmlimagelist):
    res = 0
    for txtimage in txtimagelist:
        for xmlimage in xmlimagelist:
            res += validate_img(txtimage, xmlimage)
    return res


def validate_img(img, img2):
    txtxy = img.split()
    tx1 = int(txtxy[0])
    ty1 = int(txtxy[1]) +140
    tx2 = int(txtxy[2])
    ty2 = int(txtxy[3]) +140
#    size = txtxy[4]
    img1 = tx1, ty1, tx2, ty2
    xx1, xy1, xx2, xy2 = img2
    centerpoint1 = ((tx1+tx2)/2, (ty1+ty2)/2)
    centerpoint2 = ((xx1+xx2)/2, (xy1+xy2)/2)
    if point_in_image(centerpoint1, img2) or point_in_image(centerpoint2, img1):
        res = 1
    else:
        res = 0
    return res


def point_in_image(point, image):
    px1, py1 = point
    ix1, iy1, ix2, iy2 = image
    if ix1 < px1 < ix2 and iy1 < py1 < iy2:
        return True
    return False



batches = ["first", "third"]
xmlFolderDir = "D:\\detectProject\\data\\sourceData\\XML\\xmldata\\50XML\\"
txtFolderDir = "D:\\detectProject\\data\\sourceData\\VIDEO\\ptxt\\"

GlobalAllDetectCount = 0
GlobalDetectCount = 0
GlobalRealCount = 0
GlobalHitCount = 0
for root, dirs, files in os.walk(txtFolderDir):
    for txtFileName in files:
        xmlFileDir = xmlFolderDir + "video" + txtFileName[:-5] + "\\"
        txtFilePath = txtFolderDir + txtFileName
        print txtFilePath
        videoAllDetectCount = 0
        videoDetectCount = 0
        videoRealCount = 0
        videoHitCount = 0

        with open(txtFilePath, 'r') as txtFile:
            frameCount = 0

            while True:
                line = txtFile.readline()
                if not line:
                    print "hit  /   real    /   detect  /   allDetect "
                    print str(videoHitCount) + "   /   " + str(videoRealCount) + "    /   " + str(videoDetectCount) +\
                          "    /   " + str(videoAllDetectCount)
                    break
                frameCount += 1
                txtImgList = []
                for image in line.split(","):
                    if image == '\n':
                        continue
                    txtImgList.append(image)
                detectCount = (len(txtImgList) - 1)
                videoAllDetectCount += detectCount

                if frameCount % 10 != 0:
                    continue
                xmlFilePath = xmlFileDir + "image" + txtFileName[:-5] + ".mp4" + str(frameCount) + ".xml"
                if not os.path.exists(xmlFilePath):
                    continue
                videoDetectCount += detectCount

                xmlImgList = read_xml(xmlFilePath)
                realCount = len(xmlImgList)
                hitCount = validate_list(txtImgList, xmlImgList)
                videoRealCount += realCount
                videoHitCount += hitCount
        GlobalAllDetectCount += videoAllDetectCount
        GlobalDetectCount += videoDetectCount
        GlobalRealCount += videoRealCount
        GlobalHitCount += videoHitCount



#        print txtFile[:-5]
#        print os.path.join(root, txtFile)
"""
for batch in batches:
    xmlFolderPath = xmlFolderDir + batch
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
"""