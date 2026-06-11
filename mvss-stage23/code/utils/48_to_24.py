import cv2 as cv
import os

workdir = 'D:\Desktop/NeRO-main/I-sum'
imgs = os.listdir(workdir)

for img in imgs:
    image=cv.imread(workdir+"/" + img)
    # img=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    cv.imwrite(workdir+"/" + img,image)