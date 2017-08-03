import os

trainDataPath = 'D:\\detectProject\\traindata'
file_objectS = open('D:\\detectProject\\SmallTrainData.txt', 'w')
file_objectM = open('D:\\detectProject\\MiddleTrainData.txt', 'w')
file_objectL = open('D:\\detectProject\\LargeTrainData.txt', 'w')

small, middle, large = 0, 0, 0

for root, dirs, files in os.walk(trainDataPath):
    for trainData in files:
        tdPath = str(trainData)
        if str(trainData)[0] == 'S':
            file_objectS.write(tdPath + '\n')
            small += 1
        elif str(trainData)[0] == 'M':
            file_objectM.write(tdPath + '\n')
            middle += 1
        elif str(trainData)[0] == 'L':
            file_objectL.write(tdPath + '\n')
            large += 1

print 'small: ' + str(small)
print 'middle: ' + str(middle)
print 'large: ' + str(large)
file_objectL.close()
file_objectM.close()
file_objectS.close()

