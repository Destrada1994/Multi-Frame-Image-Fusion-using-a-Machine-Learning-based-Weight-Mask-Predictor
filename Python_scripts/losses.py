import torch
import torch.nn as nn
import math
import pickle
from torch.autograd import Variable
import torchvision
import numpy as np

def FFT_loss(x,y):
    fft_x = torch.fft.fft2(x)
    fft_y = torch.fft.fft2(y) 
    
    fft_amp_x=fft_x.abs()
    fft_pha_x=fft_x.angle()
    
    fft_amp_y=fft_y.abs()
    fft_pha_y=fft_y.angle()
    
    z1=fft_amp_x-fft_amp_y
    z2=fft_pha_x-fft_pha_y
    z1=torch.abs(z1)
    z2=torch.abs(z2)
    
    return z1.mean()+z2.mean()


class loss_VGG16(nn.Module):
    def __init__(self, feature_layers=[0, 1, 2, 3]):
        super(loss_VGG16, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        
        self.feature_layers = feature_layers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
                #loss += torch.nn.functional.l1_loss(x, y)
        return loss
    
class loss_VGG19(nn.Module):
    def __init__(self, feature_layers=[0, 1, 2, 3]):
        super(loss_VGG19, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:10].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:18].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[16:26].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        
        self.feature_layers = feature_layers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
                #loss += torch.nn.functional.l1_loss(x, y)
        return loss
    
def gauss_kernel(error, sigma = 0.5):
    lambda1 = 1/(2*(sigma**2))
    G=(1/(np.sqrt(2*np.pi)*sigma))*torch.exp(-lambda1*torch.square(error))
    return G

def loss_correntropy( x,y, sigma=0.5):
    error=x-y
    G0=(1/(np.sqrt(2*np.pi)*sigma))
    Gxy=gauss_kernel(error,sigma)
    
    c_loss=G0-torch.mean(Gxy)
    return c_loss

def loss_L1(x, y, size_average=False):
  z = x - y
  z2=torch.abs(z)
  if size_average:
    return z2.mean()
  else:
    return z2.sum().div(x.size(0)*2)

def loss_MSE(x, y, size_average=False):
  z = x - y 
  z2 = z * z
  if size_average:
    return z2.mean()
  else:
    return z2.sum().div(x.size(0)*2)

def criterion_MSE(x, y):
    z = x - y
    z2 = z * z
    return z2.mean() 

def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)

class GANLoss(nn.Module):
    def __init__(self, use_l1=True):
        super(GANLoss, self).__init__()
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def forward(self,input,target):
        return self.loss(input,target)
