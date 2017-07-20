import os

trainDataPath = 'D:\\detectProject\\traindata'
file_objectS = open('D:\\detectProject\\SmallTrainData.txt', 'w')
file_objectM = open('D:\\detectProject\\MiddleTrainData.txt', 'w')
file_objectL = open('D:\\detectProject\\LargeTrainData.txt', 'w')

for root, dirs, files in os.walk(trainDataPath):
    for trainData in files:
        tdPath = os.path.join(root, trainData)
        print str(trainData)[3]
        if str(trainData)[3] == 'S':
            file_objectS.write(tdPath + '\n')
        elif str(trainData)[3] == 'M':
            file_objectM.write(tdPath + '\n')
        elif str(trainData)[3] == 'L':
            file_objectL.write(tdPath + '\n')

file_objectL.close()
file_objectM.close()
file_objectS.close()

