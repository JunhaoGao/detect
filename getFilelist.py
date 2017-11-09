import os
dataPath = 'D:\\detectProject\\data\\sourceData\\VIDEO\\50videos'
file_object = open('D:\\detectProject\\data\\sourceData\\TRAINDATA\\videoList.txt', 'w')

for root, dirs, files in os.walk(dataPath):
    for folder in files:
        ndPath = str(folder)
        file_object.write(dataPath + '\\' + ndPath + '\n')

file_object.close()