import cv2
import os

# 指定包含图片和gt图片的文件夹路径
image_folder = r'D:\Users\16590\Desktop\result-new\Test\low'
gt_folder = r'D:\Users\16590\Desktop\FLW-Net-main\data\Test\high'

# 获取文件夹中的所有文件
image_files = os.listdir(image_folder)
gt_files = os.listdir(gt_folder)

# 确保文件数量一致
if len(image_files) != len(gt_files):
    print("Error: 图片和GT图片的数量不一致")
    exit()
total_psnr = 0.0
# 逐个计算每张图片的PSNR值
for i in range(len(image_files)):
    image_path = os.path.join(image_folder, image_files[i])
    gt_path = os.path.join(gt_folder, gt_files[i])

    # 读取图片
    image = cv2.imread(image_path)
    gt = cv2.imread(gt_path)

    # 检查图片尺寸是否一致
    if image.shape != gt.shape:
        print(f"Error: 图片{image_files[i]}和GT图片{gt_files[i]}的尺寸不一致")
        continue

    # 计算PSNR值
    psnr = cv2.PSNR(gt, image)
    total_psnr += psnr

    print(f"图片{image_files[i]}的PSNR值: {psnr}")
average_psnr = total_psnr / len(image_files)
print(f"所有图片的平均PSNR值: {average_psnr}")
