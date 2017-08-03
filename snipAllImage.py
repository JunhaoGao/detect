import cv2
import os

videoPath = "D:\\detectProject\\data\\video\\first" #input your video path
for root, dirs, files in os.walk(videoPath):
    for file in files:
        dirPath = "D:\\detectProject\\data\\image\\" + file.split(".MOV")[0]
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        videoCapture = cv2.VideoCapture(os.path.join(root, file))
        fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        print os.path.join(root, file)
        success, frame = videoCapture.read()
        #frame = cv2.resize(frame, (480, 320)) #zip the frame
        c = 0
        while success:
            #frame = cv2.resize(frame, (480, 320)) #zip the frame
            cv2.imwrite(dirPath + '\\image' + str(c) + '.jpg', frame)
            success, frame = videoCapture.read()
            c = c + 1