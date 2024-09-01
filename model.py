import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pytorch_ssim
from DDEB import DDEB
from SFEM import SFEM


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False,isusePL=True):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.isusePL = isusePL
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        if self.isusePL:
            self.act = torch.nn.PReLU()
    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        if self.isusePL:
            out = self.act(out)
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class spatial_attn_layer(nn.Module):
        def __init__(self, kernel_size=5):
            super(spatial_attn_layer, self).__init__()
            self.compress = ChannelPool()
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        def forward(self, x):
            x_compress = self.compress(x)
            x_out = self.spatial(x_compress)
            scale = torch.sigmoid(x_out)  # broadcasting
            return x * scale
class HA(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding=0):
        super(HA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBlock(input_size, output_size//2, kernel_size, stride, padding, bias=True,isusePL=False)
        self.conv2 = ConvBlock(input_size+output_size//2, output_size, kernel_size, stride, padding, bias=True,isusePL=False)
        self.resize = nn.functional.interpolate
        self.spa=spatial_attn_layer()
    def forward(self, x):
        p1=self.max_pool(x)
        p2=self.avg_pool(x)
        p1=(p1+p2)/2
        p1=self.resize(p1,size=[x.size()[2],x.size()[3]],scale_factor=None, mode='nearest')
        p2=self.conv1(p1)
        p5=self.spa(p2)
        p3=torch.cat((x, p5), 1)
        p4=self.conv2(p3)
        return p4

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        return out

class CSDN_Temd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Temd, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        return out

class Hist_adjust(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Hist_adjust, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, input):
        out = self.point_conv(input)
        return out


class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor=20,nbins=14):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.LeakyReLU(inplace=True)
		self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.scale_factor = scale_factor
		self.nbins = nbins
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 16

#   RTEE-Net
		self.e_conv1 = nn.Conv2d(4, number_f, 3, padding=1, groups=1)
		self.SFEM = SFEM()
		self.e_conv2 = DDEB(16, drop_path=0.01)
		self.e_conv3 = CSDN_Tem(number_f + 3 + 1, number_f)
		self.e_conv4 = DDEB(16, drop_path=0.05)
		self.e_conv5 = DDEB(16, drop_path=0.1)
		self.e_conv6 = HA(16, 16)
		self.e_conv7 = CSDN_Tem(number_f+3,6)
#   Highe-order curve adjustment
		self.g_conv1 = Hist_adjust(self.nbins+1,number_f) 
		self.g_conv2 = Hist_adjust(number_f,number_f) 
		self.g_conv3 = Hist_adjust(number_f*2,number_f)
		self.g_conv4 = Hist_adjust(number_f+self.nbins+1,number_f)
		self.g_conv5 = Hist_adjust(number_f,7)


	def retouch(self, x,x_r):

		x = x + x_r[:,0:1,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,1:2,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,2:3,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,3:4,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,4:5,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,5:6,:,:]*(-torch.pow(x,2)+x)

		enhance_image = x + x_r[:,6:7,:,:]*(-torch.pow(x,2)+x)

		return enhance_image
		
	def forward(self, x, hist):
		SFEM = self.SFEM(x)
		x_V = x.max(1,keepdim=True)[0]
		if self.scale_factor==1:
			# x_V_down = torch.mean(x_V,[2,3],keepdim=True)
			x_V_up = torch.mean(x_V,[2,3],keepdim=True)+x_V*0
		else:
			x_V_down = F.interpolate(x_V,scale_factor=1/self.scale_factor, mode='bilinear')
			x_V_up = F.interpolate(x_V_down,scale_factor=self.scale_factor, mode='bilinear')

		g1 = self.relu(self.g_conv1(hist))
		g2 = self.relu(self.g_conv2(g1))
		g3 = self.relu(self.g_conv3(torch.cat([g2,g1],1)))
		g4 = self.relu(self.g_conv4(torch.cat([g3,hist],1)))
		g5 = self.relu(self.g_conv5(g4))

		retouch_image = self.retouch(x_V,g5)

		x1 = self.relu1(self.e_conv1(torch.cat([SFEM,x_V_up/2],1)))
		x2 = self.e_conv2(x1)
		x3 = self.e_conv3(torch.cat([x2,SFEM,retouch_image],1))
		x4 = self.e_conv4(x3)
		x5 = self.e_conv5(x4)
		x6 = self.e_conv6(x5)
		enhance_image = F.softplus(self.e_conv7(torch.cat([x6,SFEM],1)))

		return enhance_image[:,0:3,:,:],retouch_image,enhance_image[:,3:,:,:]

if __name__ == "__main__":
	net = enhance_net_nopool()
	print('total parameters:', sum(param.numel() for param in net.parameters()))