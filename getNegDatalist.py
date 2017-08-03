import os

trainDataPath = 'D:\\detectProject\\negativedata'
file_objectNeg1 = open('D:\\detectProject\\NegativeData1.txt', 'w')
file_objectNeg2 = open('D:\\detectProject\\NegativeData2.txt', 'w')
file_objectNeg3 = open('D:\\detectProject\\NegativeData3.txt', 'w')

small, middle, large = 0, 0, 0

for root, dirs, files in os.walk(trainDataPath):
    for negData in files:
        ndPath = str(negData)
        if str(negData)[0] == 'S':
            file_objectNeg1.write(ndPath + '\n')
            small += 1
        elif str(negData)[0] == 'M':
            file_objectNeg2.write(ndPath + '\n')
            middle += 1
        elif str(negData)[0] == 'L':
            file_objectNeg3.write(ndPath + '\n')
            large += 1

file_objectNeg1.close()
file_objectNeg2.close()
file_objectNeg3.close()
print 'small: ' + str(small)
print 'middle: ' + str(middle)
print 'large: ' + str(large)