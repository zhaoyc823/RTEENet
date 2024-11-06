import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import Myloss
import pytorch_ssim
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = 700000000

ssim_loss = pytorch_ssim.SSIM()
def lowlight(image_path,image_high_name,expect_mean):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 20
	nbins = 14
	exp_mean = expect_mean

	data_lowlight = Image.open(image_path)
	data_highlight = (np.asarray(Image.open(image_high_name))/255.0)
	exp_mean = np.max(data_highlight,axis=2,keepdims=True).mean()
	data_highlight = torch.from_numpy(data_highlight).float().permute(2,0,1).cuda().unsqueeze(0)

	data_lowlight = (np.asarray(data_lowlight)/255.0)
	low_im_filter_max = np.max(data_lowlight,axis=2,keepdims=True)  # positive
	hist = np.zeros([1,1,int(nbins+1)])

	xxx,bins_of_im = np.histogram(low_im_filter_max ,bins = int(nbins-2),range=(np.min(low_im_filter_max),np.max(low_im_filter_max)))
	
	hist_c = np.reshape(xxx,[1,1,nbins-2])
	hist[:, :, 0:nbins-2]  =  np.array(hist_c, dtype=np.float32)/np.sum(hist_c)
	hist[ :, :, nbins-2:nbins-1]  =  np.min(low_im_filter_max)
	hist[ :, :, nbins-1:nbins]  =  np.max(low_im_filter_max)
	hist[ :, :, -1] = exp_mean

	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	hist = torch.from_numpy(hist).float().permute(2,0,1).cuda().unsqueeze(0)

	RTEE_net = model.enhance_net_nopool(scale_factor,nbins).cuda()
	RTEE_net.load_state_dict(torch.load('snapshots_RTEEnet/best_Epoch.pth'))
	start = time.time()
	enhanced_image,retouch_image,ill_image = RTEE_net(data_lowlight,hist)
	end_time = (time.time() - start)
	# print(end_time)

	data_highlight_max = torch.mean(data_highlight,1,keepdims=True)
	data_enhanced_image_max = torch.mean(enhanced_image,1,keepdims=True)
	
	data_highlight_max = F.avg_pool2d(data_highlight_max,31,stride=1, padding=15,count_include_pad =False)
	data_enhanced_image_max = F.avg_pool2d(data_enhanced_image_max,31,stride=1, padding=15,count_include_pad =False)

	imdff = enhanced_image- data_highlight
	rmse = torch.mean(imdff**2)
	Loss_psnr = 10*torch.log10(1/rmse)
	Loss_ssim = ssim_loss(enhanced_image,data_highlight)
	image_path = image_path.replace('data','result')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
		# print(image_path.replace('/'+image_path.split("/")[-1],''))
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time,Loss_psnr,Loss_ssim

if __name__ == '__main__':

	with torch.no_grad():

		filePath = './LOL_v1/Test/low/'
		filePath_high = './LOL_v1/Test/high/'
		file_list = os.listdir(filePath)
		sum_time = 0

		import numpy as np
		#expect_mean = 0.4
		expect_mean = np.linspace(0.4,0.5,2)
		psnrs = []
		ssims = []
		for i in range(len(expect_mean)):
			psnr = 0
			ssim = 0
			for file_name in file_list:
				image_name = filePath+file_name
				image_high_name = filePath_high+file_name.replace('low','normal')
				runtime,Loss_psnr,Loss_ssim = lowlight(image_name,image_high_name,expect_mean[i])
				sum_time = sum_time + runtime
				psnr = psnr + Loss_psnr
				ssim = ssim + Loss_ssim
			psnrs.append(psnr.cpu()/len(file_list))
			ssims.append(ssim.cpu()/len(file_list))
			print(sum_time)
			print("the psnr and ssim are %s, %s "%(psnr/len(file_list),ssim/len(file_list)))
		np.save("psnr.npy",np.asarray(ssims))
		np.save("ssim.npy",np.asarray(psnrs))

