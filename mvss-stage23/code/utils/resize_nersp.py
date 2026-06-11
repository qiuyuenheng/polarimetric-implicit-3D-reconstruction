import os
from PIL import Image
import numpy as np


if __name__ == '__main__':
    workdir = "D:\Desktop\读研笔记\MVSS\dtu/LION/"
    savepath = "D:\Desktop\读研笔记\MVSS\dtu/LION/"
    DIR = ['I-sum']

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    list = os.listdir(workdir)
    for name in list:
        img = np.array(Image.open(workdir + name))
        mask = np.zeros_like(img[:,:,0])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img[i,j] != (127,127,127,0)).any():
                    mask[i,j] = 255
        m = Image.fromarray(mask)
        m.save(savepath + name)
