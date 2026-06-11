from PIL import Image

# 打开原始图片
original_image_path = 'D:\Desktop\读研期间零碎资料\论文\实验结果\对比实验\龙/dragon-gt.png'  # 替换为你的图片路径
original_image = Image.open(original_image_path)

imgsize = [800, 700]

# 计算截取区域的左上角坐标
# 因为我们要截取最中间的100x100部分，所以左上角坐标是 ((300-100)//2, (300-100)//2)
left = (original_image.size[0] - imgsize[0]) // 2
top = (original_image.size[1] - imgsize[1]) // 2
right = left + imgsize[0]
bottom = top + imgsize[1]

# 使用crop()方法截取图片
cropped_image = original_image.crop((left, top, right, bottom))

cropped_image = cropped_image.resize((imgsize[0]//2, imgsize[1]//2))

cropped_image = cropped_image.convert('RGB')

# 保存截取的图片
cropped_image_path = 'D:\Desktop\读研期间零碎资料\论文\实验结果\对比实验\龙/gt_cropped_image.jpg'  # 替换为你想要保存的路径
cropped_image.save(cropped_image_path)

print("图片截取完成，并已保存。")