import cv2
import os

videoPath = "D:\\detectProject\\data\\third" #input your video path
for root, dirs, files in os.walk(videoPath):
    for file in files:
        videoCapture = cv2.VideoCapture(os.path.join(root, file))
        fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        print os.path.join(root, file)
        success, frame = videoCapture.read()
        #frame = cv2.resize(frame, (480, 320)) #zip the frame
        c = 0
        while success:
            if ( c % 5 == 0 ):
                #frame = cv2.resize(frame, (480, 320)) #zip the frame
                cv2.imwrite('D:/detectProject/dataframe/third/' + 'image' + str(file).split('.MOV')[0] + str(c) + '.jpg',
                            frame)
            success, frame = videoCapture.read()
            c = c + 1