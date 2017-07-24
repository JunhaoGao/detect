import os

trainDataPath = 'D:\\detectProject\\negativedata'
file_objectNeg = open('D:\\detectProject\\NegativeData.txt', 'w')

num = 0
for root, dirs, files in os.walk(trainDataPath):
    for negData in files:
        ndPath = str(negData)
        file_objectNeg.write(ndPath + '\n')
        num += 1

file_objectNeg.close()
print num
