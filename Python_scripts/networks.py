import torch
import torch.nn as nn
import math
import pickle
from torch.autograd import Variable
import torchvision
import functools
import numpy as np

class WaveletTransform(nn.Module): 
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()
        
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        
        ks = int(self.scale)
        nc = 3 * ks * ks
        
        if dec:
          self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else:
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path,'rb')
                dct = pickle.load(f,encoding='latin1')
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            #print(osz)
            output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
          output = self.conv(xx)        
        return output

class DenseBlock2D(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock2D, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock2D(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='reflect', activation='relu', norm='batch'):
        super(ConvBlock2D, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias,padding_mode=padding_mode)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock3D(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='reflect',activation='relu', norm='batch'):
        super(ConvBlock3D, self).__init__()
        self.conv = torch.nn.Conv3d(input_size, output_size, kernel_size, stride, padding, bias=bias,padding_mode=padding_mode)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock3D(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(DeconvBlock3D, self).__init__()
        self.deconv = torch.nn.ConvTranspose3d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ResnetBlock3D(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect', activation='relu', norm='batch'):
        super(ResnetBlock3D, self).__init__()
        self.conv1 = torch.nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding, bias=bias,padding_mode=padding_mode)
        self.conv2 = torch.nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding, bias=bias,padding_mode=padding_mode)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm3d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm3d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out
        
class weights_generator(nn.Module):
    def __init__(self,num_channels=3, base_filter=64,num_input_images=9):
        super(weights_generator, self).__init__()
        self.num_channels=num_channels
        self.num_input_images=num_input_images
        k=3
        num_residuals=3
        
        self.layer_input=ConvBlock3D(num_channels, base_filter, (k, 7, 7), 1, (k//2, 7//2, 7//2),activation='relu', norm='batch')

        self.layer_down = ConvBlock3D(base_filter, base_filter*2, (k, 4, 4), (1, 2, 2), (k//2, 1, 1),activation='relu', norm='batch')

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock3D(base_filter*2,(k, 3, 3),1,(k//2, 1, 1), activation='relu', norm=None))
        self.residual_layers = nn.Sequential(*resnet_blocks)
        
        self.layer_up = DeconvBlock3D(base_filter*2, base_filter, (k, 4, 4), (1, 2, 2), (k//2, 1, 1),activation='relu', norm='batch')
        
        self.layer_output=ConvBlock3D(base_filter, 1, (k, 7, 7), 1, (k//2, 7//2, 7//2),activation='sigmoid', norm=None)

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        weights = self.layer_input(x)
        weights = self.layer_down(weights)
        weights = self.residual_layers(weights)
        weights = self.layer_up(weights)
        weights = self.layer_output(weights)
        weights = weights.expand((-1,self.num_channels,-1,-1,-1))
        
        out = x*weights
        out = torch.sum(out,dim=2)
        out = torch.clamp(out, min=0, max=1)
        
        return weights, out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

class discriminator(nn.Module):
    def __init__(self,num_channels=3, base_filter=64, input_height=128, input_width=None):
        super(discriminator, self).__init__()
        if input_width == None:
            input_width = input_height
        output_width=input_width
        output_height=input_height
        #image shape: 3 x 128 x 128
        self.input_layer = ConvBlock2D(num_channels, base_filter, kernel_size=4, stride=2, padding=1,activation='lrelu', norm='batch')
        #image shape: 64 x 64 x 64
        self.layer_1 = ConvBlock2D(base_filter, base_filter//2, kernel_size=4, stride=2, padding=1,activation='lrelu', norm='batch')
        #image shape: 32 x 32 x 32
        self.layer_2 = ConvBlock2D(base_filter//2, base_filter//4, kernel_size=4, stride=2, padding=1,activation='lrelu', norm='batch')
        #image shape: 16 x 16 x 16
        self.flatten=nn.Flatten()
        
        self.dense_1=DenseBlock2D((base_filter//4)*(output_width//8)*(output_height//8),1024,activation='lrelu', norm='batch')
        self.dense_2=DenseBlock2D(1024,1,activation='sigmoid', norm=None)
      
    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.dense_2(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)
