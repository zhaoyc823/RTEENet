import torch
import torch.nn as nn
import torch.nn.functional as F

channel = 32

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=((kernel_size-1)//2), bias=True,dilation=1)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class SFEM(nn.Module):
    def __init__(self):
        super(SFEM, self).__init__()

        self.baseConv1 = nn.Sequential(
                     nn.Conv2d(3,channel,5,1,padding=2,bias=True,dilation=1),
                     nn.ReLU()
        )

        self.baseConv2 = nn.Sequential(
                     nn.Conv2d(channel,3,5,1,padding=2,bias=True,dilation=1),
        )

        self.RBD1 = RDB(channel,2,int(channel/2)) 
        self.RBD2 = RDB(channel,2,int(channel/2))
        
    def forward(self, x):  
 
        bsConv1 = self.baseConv1(torch.cat([x],1))
        RBD1 = self.RBD1(bsConv1)
        RBD2 = self.RBD2(RBD1)
        base = self.baseConv2(RBD2) + x
        return base


