from PIL import Image
import numpy as np

if __name__ == '__main__':
    workdir = 'D:\Desktop\读研笔记\MVSS\dtu\dragon/'
    img_name = "I-sum/0017.png"
    mask_name = "masks/0017.png"
    img = np.array(Image.open(workdir + img_name))[:, :, :3]
    mask = np.array(Image.open(workdir + mask_name))
    new_img = img.copy()

    # th = 9

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == 0:
                new_img[i,j,:] = np.array([255,255,255])

    new_img = Image.fromarray(new_img)
    new_img.save(workdir+"0017.png")