import os
from PIL import Image

def crop_images_in_folder(folder_path, output_folder, x, y, width, height):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历文件夹中的所有PNG图片
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            input_image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder, filename)
            crop_image(input_image_path, output_image_path, x, y, width, height)

def crop_image(input_image_path, output_image_path, x, y, width, height):
    # 打开输入图片
    image = Image.open(input_image_path)
    
    # 裁剪图像
    cropped_image = image.crop((x, y, x+width, y+height))
    
    # # 转换为灰度图像
    # grayscale_image = cropped_image.convert("L")
    # 缩放图像
    resized_image = cropped_image.resize((300, 450))
    
    # 保存裁剪后的图像
    resized_image.save(output_image_path)

# 当前文件夹路径
folder_path = './data/wave8/'  # 可以根据实际情况更改

# 输出文件夹路径
output_folder = './data/wave8/wave8/'  # 可以根据实际情况更改

# 指定裁剪区域的位置和大小
x = 430  # 起始横坐标
y = 5  # 起始纵坐标
width = 600  # 裁剪宽度
height = 900  # 裁剪高度

# 裁剪当前文件夹下的所有PNG图片
crop_images_in_folder(folder_path, output_folder, x, y, width, height)
