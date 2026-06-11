from PIL import Image
import numpy as np
import itertools

def temp(normal):
    # 找到底是哪个坐标系

    # 生成所有由'0'和'1'组成的四位排列组合
    combinations = [''.join(p) for p in itertools.product('01', repeat=3)]

    for i in combinations:
        new_normal = np.zeros_like(normal)

        new_normal[:, :, 0] = (-1) ** int(i[0]) * normal[:, :, 0]
        new_normal[:, :, 1] = (-1) ** int(i[1]) * normal[:, :, 1]
        new_normal[:, :, 2] = (-1) ** int(i[2]) * normal[:, :, 2]

        new_normal = Image.fromarray(new_normal)
        new_normal.save("./kk/pandora_{0}.png".format(i))


def pandora(normal, mask):
    new_normal = np.zeros_like(normal)
    new_normal[:, :, 0] = -normal[:, :, 0]
    new_normal[:, :, 1] = normal[:, :, 1]
    new_normal[:, :, 2] = -normal[:, :, 2]

    for i in range(normal.shape[0]):
        for j in range(normal.shape[1]):
            if mask[i, j] == 0:
                new_normal[i, j] = np.array([0, 0, 0])

    new_normal = Image.fromarray(new_normal)
    new_normal.save("./pandora.png")

if __name__ == '__main__':
    # normal = np.array(Image.open("D:\Desktop\读研笔记\TransPIR-main10\coode\evaluation\david/pandora.png"))
    # mask = np.array(Image.open("D:\Desktop\读研笔记\TransPIR-main10\coode\evaluation/mask-hedgehog0033.png"))
    normal = np.array(Image.open("./normal_3250.png"))
    # mask = np.array(Image.open("D:\Desktop\读研笔记\TransPIR-main10\coode\evaluation/mask-hedgehog0033.png"))
    new_normal = np.zeros_like(normal)

    temp(normal)

    # pandora(normal,mask)



