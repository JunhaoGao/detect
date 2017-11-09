import os
import cv2

trainDataPath = 'D:\\detectProject\\data\\sourceData\\TRAINDATA\\man'
file_objectS = open('D:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallTrainData.txt', 'w')
file_objectM = open('D:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleTrainData.txt', 'w')
file_objectL = open('D:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeTrainData.txt', 'w')

small, middle, large = 0, 0, 0



for root, dirs, files in os.walk(trainDataPath):
    for trainData in files:
        tdPath = str(trainData)
        image = cv2.imread(trainDataPath + '\\' + tdPath)

        if image.shape[0] == 30:
            file_objectS.write(trainDataPath + '\\' + tdPath + '\n')
            small += 1
        elif image.shape[0] == 50:
            file_objectM.write(trainDataPath + '\\' + tdPath + '\n')
            middle += 1
        elif image.shape[0] == 100:
            file_objectL.write(trainDataPath + '\\' + tdPath + '\n')
            large += 1

print 'small: ' + str(small)
print 'middle: ' + str(middle)
print 'large: ' + str(large)
file_objectL.close()
file_objectM.close()
file_objectS.close()

