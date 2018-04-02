import numpy as np
import cv2

def EM(pModel, width, height):
    sum = np.double(0.0)
    for i in range(0,height):
        for j in range(0,width):
            sum += pModel[i][j]
    return sum

def EM2(pModel, width, height):
    sum = np.double(0.0)
    for i in range(0,height):
        for j in range(0,width):
            sum += pModel[i][j]*1.0*pModel[i][j]
    return sum

def EI(pToSearch, l, h, u, v, pModel, width, height):
    sum = np.double(0.0)
    roi = pToSearch[v:v+height, u:u+width]
    for i in range(0,height):
        for j in range(0,width):
            sum += roi[i][j]
    return sum

def EI2(pToSearch, l, h, u, v, pModel, width, height):
    sum = np.double(0.0)
    roi = pToSearch[v:v+height, u:u+width]
    for i in range(0,height):
        for j in range(0,width):
            sum += roi[i][j]*1.0*roi[i][j]
    return sum

def EIM(pToSearch, l, h, u, v, pModel, width, height):
    sum = np.double(0.0)
    roi = pToSearch[v:v+height, u:u+width]
    for i in range(0,height):
        for j in range(0,width):
            sum += pModel[i][j]*1.0*roi[i][j]
    return sum

def Match(pToSearch, l, h, pModel, width, height):
    uMax = l-width
    vMax = h-height
    N = width*height
    len = (uMax+1)*(vMax+1)
    MatchRec = [0.0 for x in range(0, len)]
    k = 0

    M = EM(pModel,width,height)
    M2 = EM2(pModel,width,height)
    for p in range(0, uMax+1):
        for q in range(0, vMax+1):
            I = EI(pToSearch,l,h,p,q,pModel,width,height)
            I2 = EI2(pToSearch,l,h,p,q,pModel,width,height)
            IM = EIM(pToSearch,l,h,p,q,pModel,width,height)

            numerator=(N*IM-I*M)*(N*IM-I*M)
            denominator=(N*I2-I*I)*(N*M2-M*M)

            ret = numerator/denominator
            MatchRec[k]=ret
            k+=1

    val = 0
    k = 0
    x = y = 0
    for p in range(0, uMax+1):
        for q in range(0, vMax+1):
            if MatchRec[k] > val:
                val = MatchRec[k]
                x = p
                y = q
            k+=1
    print "val: %f"%val
    return (x, y)

def main():
    img = cv2.imread('niu.jpg', cv2.IMREAD_GRAYSCALE)
    temp = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)

    print temp.shape
    imgHt, imgWd = img.shape
    tempHt, tempWd = temp.shape
    #print EM(temp, tempWd, tempHt)
    (x, y) = Match(img, imgWd, imgHt, temp, tempWd, tempHt)
    cv2.rectangle(img, (x, y), (x+tempWd, y+tempHt), (0,0,0), 2)
    cv2.imshow("temp", temp)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()