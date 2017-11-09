import cv2.cv as cv


def Sharp(image, flag1=0, flag2=0):
    w = image.width
    h = image.height
    size = (w, h)
    iSharp = cv.CreateImage(size, 8, 1)
    for i in range(h - 1):
        for j in range(w - 1):
            if flag2 == 0:
                x = abs(image[i, j + 1] - image[i, j])
                y = abs(image[i + 1, j] - image[i, j])
            else:
                x = abs(image[i + 1, j + 1] - image[i, j])
                y = abs(image[i + 1, j] - image[i, j + 1])
            if flag1 == 0:
                iSharp[i, j] = max(x, y)
            else:
                iSharp[i, j] = x + y
    return iSharp

#image = cv.LoadImage('1494839078.jpg', 0)
iMaxSharp = Sharp(image)
iAddSharp = Sharp(image, 1)
iRMaxSharp = Sharp(image, 0, 1)
iRAddSharp = Sharp(image, 1, 1)
cv.ShowImage('iMaxSharp', iMaxSharp)
cv.ShowImage('image', image)
cv.ShowImage('iAddSharp', iAddSharp)
cv.ShowImage('iRAddSharp', iRAddSharp)
cv.ShowImage('iRMaxSharp', iRMaxSharp)
cv.WaitKey(0)
