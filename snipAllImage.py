import cv2
import os
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip as vfc


videoPathList = ["D:\\detectProject\\data\\video\\50videos"]#input your video path
for videoPath in videoPathList:
    print videoPath
    for root, dirs, files in os.walk(videoPath):
        for file in files:
            dirPath = "D:\\detectProject\\data\\sourceData\\IMAGE4\\" + file.split(".MOV")[0]
            if not os.path.exists(dirPath):
                os.mkdir(dirPath)
            print os.path.join(root, file)
            clip = vfc(os.path.join(root, file))
            print dirPath+ '\\image%04d.jpg'
            clip.write_images_sequence(dirPath+ '\\image%04d.jpg', fps=1.498)
            #videoCapture = cv2.VideoCapture(os.path.join(root, file))
            #fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
            #size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            #        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
            #success, frame = videoCapture.read()
            #frame = cv2.resize(frame, (480, 320)) #zip the frame
            #c = 0
            #while success:
                #frame = cv2.resize(frame, (480, 320)) #zip the frame
            #    cv2.imwrite(dirPath + '\\image' + str(c) + '.jpg', frame)
            #    success, frame = videoCapture.read()
            #    c = c + 1