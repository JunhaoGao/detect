#coding:utf-8
import cv2
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

imagePath = "C:\\Users\\GarenGao\\Desktop\\2017科学营\\营员及带队教师照片" #input your video path
imagePath = unicode(imagePath, "utf-8")
print(os.walk(imagePath))
wnd = 'OpenCV Video'
for root, dirs, files in os.walk(imagePath):
    for file in files:
        #print os.path.join(root, file)
        path1 = str(os.path.join(root, file))
        print path1
path = 'D:/新建文件夹/pic.jpg'
# path.encode('utf-8')
path = unicode(path, "utf8").encode('gbk')
image = cv2.imread(path, 3)
cv2.imshow("test", image)
cv2.waitKey(0)
        #image = cv2.resize(image, (480, 320))
        #cv2.imwrite(imagePath + '/image/' + str(file) + '1.jpg', image)
        #print"success"



        #success, frame = videoCapture.read()
        #frame = cv2.resize(frame, (480, 320)) #zip the frame
        #c = 0
        #while success:
        #    if ( c % 5 == 0 ):
        #        #frame = cv2.resize(frame, (480, 320)) #zip the frame
        #        cv2.imwrite('image/'+ str(file) + str(c) + '.jpg', frame)
        #    success, frame = videoCapture.read()
        #    c = c + 1