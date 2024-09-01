import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision.transforms.functional import to_tensor
from pytorch_ssim import ssim

def calculate_ssim(img1, img2):
    img1 = Variable(to_tensor(img1).unsqueeze(0))
    img2 = Variable(to_tensor(img2).unsqueeze(0))
    return ssim(img1, img2)

def calculate_ssim_folder(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    total_ssim = 0.0
    num_comparisons = 0

    for file1 in files1:
        if file1 in files2:
            img1_path = os.path.join(folder1, file1)
            img2_path = os.path.join(folder2, file1)

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            current_ssim = calculate_ssim(img1, img2)
            total_ssim += current_ssim
            num_comparisons += 1

    if num_comparisons == 0:
        print("No common files found for comparison.")
        return 0.0

    average_ssim = total_ssim / num_comparisons
    return average_ssim.item()

if __name__ == "__main__":
    folder1_path = r'D:\Users\16590\Desktop\result-new\Test\low'
    folder2_path = r'D:\Users\16590\Desktop\FLW-Net-main\data\Test\high'

    average_ssim_value = calculate_ssim_folder(folder1_path, folder2_path)

    print(f"Average SSIM: {average_ssim_value}")
