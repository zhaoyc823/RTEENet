import os
from skimage import io
from lpips import LPIPS
import torch

# 设置文件夹路径
folder_path1 = r"D:\Users\16590\Desktop\result-new\Test\low"
folder_path2 = r"D:\Users\16590\Desktop\FLW-Net-main\data\Test\high"

# 获取两个文件夹中所有图像文件的路径
image_paths1 = [os.path.join(folder_path1, filename) for filename in os.listdir(folder_path1) if filename.endswith(('.jpg', '.png'))]
image_paths2 = [os.path.join(folder_path2, filename) for filename in os.listdir(folder_path2) if filename.endswith(('.jpg', '.png'))]

# 初始化变量
total_lpips = 0.0

# 加载LPIPS模型
lpips_model = LPIPS(net='alex')

# 遍历图像文件并计算LPIPS
for path1, path2 in zip(image_paths1, image_paths2):
    # 读取图像
    image1 = io.imread(path1)
    image2 = io.imread(path2)

    # 转换图像为PyTorch张量
    image_tensor1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image_tensor2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    # 标准化图像张量值到[0, 1]范围
    image_tensor1 /= 255.0
    image_tensor2 /= 255.0

    # 计算LPIPS
    lpips_score = lpips_model(image_tensor1.unsqueeze(0), image_tensor2.unsqueeze(0))

    # 累积总LPIPS
    total_lpips += lpips_score.item()

# 计算平均LPIPS值
average_lpips = total_lpips / len(image_paths1)

print(f"Total LPIPS: {total_lpips}")
print(f"Average LPIPS: {average_lpips}")
