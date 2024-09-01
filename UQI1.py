import os
from skimage import io
from torchvision import transforms
from piq import ssim

# 设置文件夹路径
folder_path1 = r"D:\Users\16590\Desktop\result-new\Test\low"
folder_path2 = r"D:\Users\16590\Desktop\FLW-Net-main\data\Test\high"

# 获取两个文件夹中所有图像文件的路径
image_paths1 = [os.path.join(folder_path1, filename) for filename in os.listdir(folder_path1) if filename.endswith(('.jpg', '.png'))]
image_paths2 = [os.path.join(folder_path2, filename) for filename in os.listdir(folder_path2) if filename.endswith(('.jpg', '.png'))]

# 初始化变量
total_uqi = 0.0

# 遍历图像文件并计算UQI
for path1, path2 in zip(image_paths1, image_paths2):
    # 读取图像
    image1 = io.imread(path1)
    image2 = io.imread(path2)

    # 转换图像为PyTorch张量
    image_tensor1 = transforms.ToTensor()(image1)
    image_tensor2 = transforms.ToTensor()(image2)

    # 计算UQI（实际上是 SSIM，piq 库中用 SSIM 来计算图像质量）
    uqi_value = ssim(image_tensor1.unsqueeze(0), image_tensor2.unsqueeze(0))

    # 累积总UQI
    total_uqi += uqi_value.item()

# 计算平均UQI值
average_uqi = total_uqi / len(image_paths1)

print(f"Total UQI (SSIM): {total_uqi}")
print(f"Average UQI (SSIM): {average_uqi}")




