import os
import cv2
import torch
from skimage import color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor


# 设置文件夹路径
folder_path = r"D:\Users\16590\Desktop\result-new\Test\low"
folder_path1 = r"D:\Users\16590\Desktop\FLW-Net-main\data\Test\high"
# 定义转换函数，将图像转换为Lab颜色空间
def rgb_to_lab(image):
    image_lab = color.rgb2lab(image)
    return torch.from_numpy(image_lab).permute(2, 0, 1).float()

# 获取文件夹中所有图像文件的路径
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png'))]
image_paths1 = [os.path.join(folder_path1, filename) for filename in os.listdir(folder_path1) if filename.endswith(('.jpg', '.png'))]
# 初始化变量
delta_e_values = []

# 遍历图像文件并计算CIEDE2000
for i in range(len(image_paths) - 1):
    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths1[i])

    # 转换图像到Lab颜色空间
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)

    # 将PyTorch张量转换为colormath中的LabColor对象
    color1 = LabColor(lab1[0, :, :].mean().item(), lab1[1, :, :].mean().item(), lab1[2, :, :].mean().item())
    color2 = LabColor(lab2[0, :, :].mean().item(), lab2[1, :, :].mean().item(), lab2[2, :, :].mean().item())

    # 计算CIEDE2000
    delta_e = delta_e_cie2000(color1, color2)

    # 将CIEDE2000值添加到列表中
    delta_e_values.append(delta_e)

# 计算平均CIEDE2000值
average_delta_e = sum(delta_e_values) / len(delta_e_values)

print(f"Total CIEDE2000: {sum(delta_e_values)}")
print(f"Average CIEDE2000: {average_delta_e}")

