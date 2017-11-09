import os
dataPath = 'D:\\detectProject\\data\\image'
file_object = open('D:\\detectProject\\data\\fileFolder.txt', 'r')

lines = file_object.readlines()
for line in lines:
    folder = line.split('\n')[0]
    fileImagePath = "D:\\detectProject\\data\\" + folder + ".txt"
    file_object1 = open(fileImagePath, 'w')
    for root, dirs, files in os.walk(dataPath+"\\"+folder):
        linenum = files.__len__()
    file_object1.write(str(linenum) + '\n')
    file_object1.close()
file_object.close()
