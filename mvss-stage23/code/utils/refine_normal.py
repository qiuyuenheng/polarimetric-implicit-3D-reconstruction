from PIL import Image
import numpy as np

def refine_neisf_data():
    """
    使用neisf的数据集产生的法向量有些缺漏，这个函数可以补齐。
    """
    workdir = "D:\Desktop\读研笔记\MVSS\dtu/bunny/"
    gt_normal = workdir + "normals/img_023.png"
    pred_normal = "D:\Desktop/normal_3550.png"
    mask = workdir + "masks/img_023.png"

    gt_normal = np.array(Image.open(gt_normal))[:, :, :3]
    pred_normal = np.array(Image.open(pred_normal))
    mask = np.array(Image.open(mask)) / 255
    new_normal = pred_normal.copy()

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # if pred_normal[i,j].sum() == 0 and mask[i,j] != 0 and gt_normal[i,j].sum() != 381:
            if pred_normal[i, j].sum() == 0 and mask[i, j].sum() != 0 and gt_normal[i, j].sum() != 0:
                new_normal[i, j] = gt_normal[i, j]

    new_img = Image.fromarray(new_normal)
    new_img.save("D:\Desktop/new.png")


if __name__ == '__main__':
    pred_normal = "./mvss_normal.png"
    mask = "D:\Desktop\读研笔记\MVSS\dtu/360ballcup2\masks/013_pseudo.png"

    pred_normal = np.array(Image.open(pred_normal))[:, :, :3]
    mask = np.array(Image.open(mask)) / 255
    refined_normal = pred_normal.copy()

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if pred_normal[i, j].sum() == 0 and mask[i, j].sum() != 0:
                refined_normal[i,j] = pred_normal[i+1,j+1]

    new_img = Image.fromarray(refined_normal)
    new_img.save("./mvss_normal_refined.png")
