import os

dataPath = "C:\\Users\\GarenGao\\Desktop\\data\\dataFromVideo\\fwd_force"
#lat_gpsPath = "C:\\Users\\GarenGao\\Desktop\\data\\dataFromVideo\\lat_gpsData.csv"
#lon_gpsPath = "C:\\Users\\GarenGao\\Desktop\\data\\dataFromVideo\\lon_gpsData.csv"
#lat_File = open(lat_gpsPath, 'w')
#lon_File = open(lon_gpsPath, 'w')
speedPath = "C:\\Users\\GarenGao\\Desktop\\data\\dataFromVideo\\fwd_forceData.csv"
speed_File = open(speedPath, 'w')

#for root, dirs, files in os.walk(dataPath):
#    for dataFile in files:
#        filePath = os.path.join(root, dataFile)
#        file_Obj = open(filePath, 'r')
#        lines = file_Obj.readlines()
#        eventNum = dataFile.split("Event")[1].split(".dce.txt")[0]
#        #lat_File.write(eventNum)
#        #lon_File.write(eventNum)
#        for line in lines:
#            if not line.__contains__("None"):
#                if line.__contains__("lat"):
#                    lat_File.write("," + line.split(" ")[1].split(",N\n")[0])
#                if line.__contains__("lon"):
#                    lon_File.write("," + line.split(" ")[1].split(",E\n")[0])#
#        lat_File.write("\n")
#        lon_File.write("\n")
#lat_File.close()
#lon_File.close()

for root, dirs, files in os.walk(dataPath):
    for dataFile in files:
        filePath = os.path.join(root, dataFile)
        file_Obj = open(filePath, 'r')
        lines = file_Obj.readlines()
        eventNum = dataFile.split("Event")[1].split(".dce.txt")[0]
        speed_File.write(eventNum)
        for line in lines:
            if not line.__contains__("None"):
                speed_File.write("," + line.split('\n')[0])#
        speed_File.write("\n")
speed_File.close()
