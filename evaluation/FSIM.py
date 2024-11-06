import os
from skimage import io
from torchvision import transforms
import torch
from piq import fsim

# 设置文件夹路径
folder_path1 = r"D:\Users\16590\Desktop\result-new\Test\low"
folder_path2 = r"D:\Users\16590\Desktop\FLW-Net-main\data\Test\high"

# 获取两个文件夹中所有图像文件的路径
image_paths1 = [os.path.join(folder_path1, filename) for filename in os.listdir(folder_path1) if filename.endswith(('.jpg', '.png'))]
image_paths2 = [os.path.join(folder_path2, filename) for filename in os.listdir(folder_path2) if filename.endswith(('.jpg', '.png'))]

# 初始化变量
total_fsim = 0.0

# 设置图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 遍历图像文件并计算FSIM
for path1, path2 in zip(image_paths1, image_paths2):
    # 读取图像
    image1 = io.imread(path1)
    image2 = io.imread(path2)

    # 转换图像为PyTorch张量
    image_tensor1 = transform(image1).unsqueeze(0)
    image_tensor2 = transform(image2).unsqueeze(0)

    # 计算FSIM
    fsim_value = fsim(image_tensor1, image_tensor2)

    # 累积总FSIM
    total_fsim += fsim_value.item()

# 计算平均FSIM值
average_fsim = total_fsim / len(image_paths1)

print(f"Total FSIM: {total_fsim}")
print(f"Average FSIM: {average_fsim}")

