# import cv2
# vc = cv2.VideoCapture('test.mp4')
# c = 1
# if vc.isOpened():
#     rval, frame = vc.read()
# else:
#     rval = False
#     print rval
# timeF = 1000
# while rval:
#     rval, frame = vc.read()
#     if(c%timeF == 0):
#         cv2.imwrite('image/'+str(c) + '.jpg',frame)
#     c = c + 1
#     cv2.waitKey(1)
# vc.release()

import cv2

wnd = 'OpenCV Video'



videoCapture = cv2.VideoCapture('D:/FFOutput/Thefirst/FILE0006.avi')

fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

cv2.namedWindow(wnd, flags=0)
cv2.resizeWindow(wnd, size[0] / 2, size[1] / 2)

success, frame = videoCapture.read()
c = 1
while success:
    cv2.imshow(wnd, frame)
    cv2.waitKey(1000 / int(fps))
    if (c % 1000 == 0):
        cv2.imwrite('image/' + str(c) + '.jpg', frame)
    success, frame = videoCapture.read()
    c = c + 1
cv2.waitKey(0)