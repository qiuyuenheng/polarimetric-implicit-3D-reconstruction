from PIL import Image
import numpy as np

if __name__ == '__main__':
    workdir = 'E:\8_5_1\8_5_1/360ballcup2/'
    img_name = "mvss-normal.png"
    img = np.array(Image.open(workdir + img_name))
    alpha = img[:,:,3]
    img = img[:,:,:3]
    new_img = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if alpha[i,j] == 0:
                new_img[i,j,:] = np.array([255,255,255])
