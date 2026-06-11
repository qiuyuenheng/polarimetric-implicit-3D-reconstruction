from PIL import Image
import numpy as np
# import cv2 as cv
import os


def simulate_48bit_image(input_image_path, output_image_path):
    # 打开图像
    img = Image.open(input_image_path)

    # 将图像转换为NumPy数组以便处理
    img_np = np.array(img)

    # 模拟增加位深度（这里只是简单地将每个颜色通道的值乘以256，实际上并没有增加颜色精度）
    # 注意：这可能会导致数据溢出，这里只是为了演示
    img_np_simulated_48bit = np.clip((img_np * 256).astype(np.uint16), 0, 65535)

    # 将处理后的NumPy数组转换回PIL图像
    img_simulated_48bit = Image.fromarray(img_np_simulated_48bit.astype(np.uint16))

    # 保存图像（注意：这里以16位深度保存，因为PIL/Pillow不支持直接保存为48位深度）
    # 使用PNG格式，因为它支持无损压缩和16位深度
    img_simulated_48bit.save(output_image_path, 'PNG')


workdir = 'D:\Desktop/ttt/I-sum/'
output_path = 'D:\Desktop/ttt/image48/'
imgs = os.listdir(workdir)

for img in imgs:
    simulate_48bit_image(workdir+img, output_path+img)

# # 使用函数
# input_image = 'path_to_your_24bit_image.jpg'
# output_image = 'simulated_48bit_image.png'
# simulate_48bit_image(input_image, output_image)