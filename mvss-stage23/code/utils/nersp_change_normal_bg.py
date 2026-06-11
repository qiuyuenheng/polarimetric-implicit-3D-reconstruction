from PIL import Image
import numpy as np

if __name__ == '__main__':
    workdir = "D:\Desktop\读研期间零碎资料\论文\实验结果\对比实验\龙/"
    img = np.array(Image.open(workdir + "nersp_6w.png"))
    new_img = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] == 128 and img[i,j,1] == 128 and img[i,j,2] == 128:
                new_img[i,j] = np.array([0,0,0])

    new_img = Image.fromarray(new_img)
    new_img.save("./nersp_normal.png")