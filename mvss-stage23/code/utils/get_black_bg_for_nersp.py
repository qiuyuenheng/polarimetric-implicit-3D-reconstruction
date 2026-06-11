from PIL import Image
import numpy as np

if __name__ == '__main__':
    workdir = 'D:\Desktop\读研期间零碎资料\论文\实验结果\对比实验/360ballcup2/'
    img_name = "mvss_normal_refined.png"
    img = np.array(Image.open(workdir + img_name))[:, :, :3]
    new_img = img.copy()

    # th = 9

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
                new_img[i,j,:] = np.array([255,255,255])
            if img[i,j,0] < img[i,j,1] and img[i-3,j,0] > img[i-3,j,1]:
                new_img[i, j, :] = np.array([255, 255, 255])

    new_img = Image.fromarray(new_img)
    new_img.save(workdir+"mvss_normal_wbg.png")