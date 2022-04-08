from __future__ import print_function
import os
import argparse
import random
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms 
import torchvision.transforms as transforms
from os import listdir, makedirs
from os.path import exists, join, basename, isfile
from math import log10
from networks import *
from losses import *
import numpy as np
import cv2
from datasets import *
from DFT_register import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enables test during training')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=4)
parser.add_argument('--dataroot', default="images/train", type=str, help='path to dataset')
parser.add_argument('--testroot', default="images/test", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=500)
parser.add_argument('--testsize', type=int, help='number of testing data', default=10)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=10, help='test batch size')
parser.add_argument('--save_iter', type=int, default=50, help='the interval iterations for saving models')
parser.add_argument('--cdim', type=int, default=1, help='the channel-size  of the input image to network')
parser.add_argument('--input_height', type=int, default=64, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=64, help='the width  of the input image to network')
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='results/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained_model", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--pretrained_model_dis", default="", type=str, help="path to pretrained Discriminator model (default: none)")
parser.add_argument("--loss", type=int, default=4, help="loss type")
parser.add_argument("--loss_2", type=int, default=0, help="loss 2 type")
parser.add_argument("--loss_2_weight", type=int, default=1, help="loss 2 weight")
parser.add_argument("--loss_dis", type=int, default=2, help="loss type")
parser.add_argument("--N_input_images", type=int, default=9, help="upsample scale")
parser.add_argument("--num_dis_updates", type=int, default=5, help="Number of Discriminator updates")
parser.add_argument("--adv_loss_weight", type=float, default=0.01, help="loss type")


    
parser.add_argument('--valid_dir', type=str, default='images_aligned/test_full_HR', help='valid image path to use')
parser.add_argument('--valid_size', type=int, default=100, help='Number valid images')
parser.add_argument('--model_path', type=str, default='model/Weight_Generator_L1Image_Discrminator_WGANGP/epoch_200.pth', help='model file to use')
parser.add_argument('--output_dir', type=str,default='valid_results_L1', help='where to save the output image')
opt = parser.parse_args()

print(opt)



if not exists(opt.output_dir):
    makedirs(opt.output_dir)

#--------------build models--------------------------

model_name='Generator'
srnet = weights_generator(num_channels=opt.cdim, base_filter=32,num_input_images=opt.N_input_images)

if os.path.isfile(opt.model_path):
    print("=> loading model '{}'".format(opt.model_path))
    weights = torch.load(opt.model_path)
    pretrained_dict = weights['model'].state_dict()
    model_dict = srnet.state_dict()
    # print(model_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    srnet.load_state_dict(model_dict)
    # srnet.load_state_dict(weights['model'].state_dict())
else:
    print("=> no model found at '{}'".format(opt.model_path))
    
#print(srnet)
if opt.cuda:
    srnet = srnet.cuda()
save_dir=join(opt.output_dir,model_name)

if not exists(save_dir):
    makedirs(save_dir)

valid_list_A, valid_list_B = load_list(opt.valid_dir, opt.valid_size)
    
idx=0
MSE=0
for i in range(opt.valid_size):
    A_filename=valid_list_A[i]
    B_filename=valid_list_B[i]
    name=basename(A_filename)
    name=name.split('.')[0]

    lr_scan, hr_img = load_valid_image(A_filename,B_filename,opt.N_input_images)

    valid_width = lr_scan.shape[2]
    valid_height = lr_scan.shape[3]
    
    stride=16
    weights=np.zeros((opt.cdim,opt.N_input_images,valid_width,valid_height))
    predict=np.zeros((opt.cdim,valid_width,valid_height))

    hr_img = hr_img[:,:valid_width, :valid_height]
         
    scan = np.array([lr_scan[:,:,valid_width-opt.input_width:,valid_height-opt.input_height:]])
        
    input = torch.from_numpy(scan)
    if opt.cuda:
        input = input.cuda()
                
    predict_weights, predict_hr = srnet(input)
    
    predict_weights=predict_weights.cpu()
    predict_hr = predict_hr.cpu()
    predict_weights=predict_weights[0].detach().numpy().astype(np.float32)
    predict_hr=predict_hr[0].detach().numpy().astype(np.float32)
      
    weights[:,:,valid_width-opt.input_width:,valid_height-opt.input_height:]=predict_weights
    predict[:,valid_width-opt.input_width:,valid_height-opt.input_height:]=predict_hr
    
    for y in range(0,valid_height-opt.input_height,stride):
        scan = np.array([lr_scan[:,:,valid_width-opt.input_width:,y:y+opt.input_height]])

        input = torch.from_numpy(scan)
        if opt.cuda:
            input = input.cuda()
                    
        predict_weights, predict_hr = srnet(input)
        
        predict_weights=predict_weights.cpu()
        predict_hr = predict_hr.cpu()
        predict_weights=predict_weights[0].detach().numpy().astype(np.float32)
        predict_hr=predict_hr[0].detach().numpy().astype(np.float32)

        if y==0:
            start_y=0
            stop_y=opt.input_height-stride
        else:
            start_y=stride
            stop_y=opt.input_height-stride
            
        weights[:,:,valid_width-opt.input_width:,y+start_y:y+stop_y]=predict_weights[:,:,:,start_y:stop_y]
        predict[:,valid_width-opt.input_width:,y+start_y:y+stop_y]=predict_hr[:,:,start_y:stop_y]

    for x in range(0,valid_width-opt.input_width,stride):
        scan = np.array([lr_scan[:,:,x:x+opt.input_width,valid_height-opt.input_height:]])

        input = torch.from_numpy(scan)
        if opt.cuda:
            input = input.cuda()
                    
        predict_weights, predict_hr = srnet(input)
        
        predict_weights=predict_weights.cpu()
        predict_hr = predict_hr.cpu()
        predict_weights=predict_weights[0].detach().numpy().astype(np.float32)
        predict_hr=predict_hr[0].detach().numpy().astype(np.float32)

        if x==0:
            start_x=0
            stop_x=opt.input_width-stride
        else:
            start_x=stride
            stop_x=opt.input_width-stride
            
            
        weights[:,:,x+start_x:x+stop_x,valid_height-opt.input_height:]=predict_weights[:,:,start_x:stop_x,:]
        predict[:,x+start_x:x+stop_x,valid_height-opt.input_height:]=predict_hr[:,start_x:stop_x,:]

    for x in range(0,valid_width-opt.input_width,stride):
        for y in range(0,valid_height-opt.input_height,stride):
            scan = np.array([lr_scan[:,:,x:x+opt.input_width,y:y+opt.input_height]])

            input = torch.from_numpy(scan)
            if opt.cuda:
                input = input.cuda()
                        
            predict_weights, predict_hr = srnet(input)
            
            predict_weights=predict_weights.cpu()
            predict_hr = predict_hr.cpu()
            predict_weights=predict_weights[0].detach().numpy().astype(np.float32)
            predict_hr=predict_hr[0].detach().numpy().astype(np.float32)

            if x==0:
                start_x=0
                stop_x=opt.input_width-stride
            else:
                start_x=stride
                stop_x=opt.input_width-stride
            if y==0:
                start_y=0
                stop_y=opt.input_height-stride
            else:
                start_y=stride
                stop_y=opt.input_height-stride
                
            weights[:,:,x+start_x:x+stop_x,y+start_y:y+stop_y]=predict_weights[:,:,start_x:stop_x,start_y:stop_y]
            predict[:,x+start_x:x+stop_x,y+start_y:y+stop_y]=predict_hr[:,start_x:stop_x,start_y:stop_y]
    
    output_filename=join(save_dir,'Full_SR_'+model_name+'_'+name+'.png')
    

    img_lr=np.zeros((opt.cdim,valid_width*opt.N_input_images,valid_height))
    img_weight=np.zeros((opt.cdim,valid_width*opt.N_input_images,valid_height))
    for ind in range(opt.N_input_images):
        lr_img=lr_scan[:,ind,:valid_width, :valid_height]
        weight_img=weights[:,ind,:,:]
        img_lr[:,valid_width*ind:valid_width*ind+valid_width,0:valid_height]=lr_img
        img_weight[:,valid_width*ind:valid_width*ind+valid_width,0:valid_height]=weight_img

    result_img=np.zeros(((opt.cdim,valid_width*opt.N_input_images,valid_height*4)))
    result_img[:,0:valid_width*opt.N_input_images,0:valid_height]=img_lr
    result_img[:,0:valid_width*opt.N_input_images,valid_height:2*valid_height]=img_weight
    result_img[:,0:valid_width,valid_height*2:3*valid_height]=predict
    result_img[:,0:valid_width,valid_height*3:4*valid_height]=hr_img
    
    result_img=result_img*255
    result_img=np.squeeze(result_img)
    result_img=Image.fromarray(result_img.astype(np.uint8))

    result_img.save(output_filename)

    target_image=hr_img*255
    target_image=np.squeeze(target_image)
    target_image=Image.fromarray(target_image.astype(np.uint8))
    input_image=img_lr[:,0:valid_width,0:valid_height]*255
    input_image=np.squeeze(input_image)
    input_image=Image.fromarray(input_image.astype(np.uint8))
    output_image=predict*255
    output_image=np.squeeze(output_image)
    output_image=Image.fromarray(output_image.astype(np.uint8))

    weights_image=img_weight*255
    weights_image=np.squeeze(weights_image)
    weights_image=Image.fromarray(weights_image.astype(np.uint8))

    target_name=join(save_dir,'HR_'+model_name+'_'+name+'.png')
    input_name=join(save_dir,'LR_'+model_name+'_'+name+'.png')
    output_name=join(save_dir,'SR_'+model_name+'_'+name+'.png')
    weights_name=join(save_dir,'Weights_'+model_name+'_'+name+'.png')

    target_image.save(target_name)
    input_image.save(input_name)
    output_image.save(output_name)
    weights_image.save(weights_name)
    
    img_hr=np.array(hr_img)
    img_out=np.array(predict)
    MSE_Loss=np.mean(np.square(img_out-img_hr))
    psnr=10 * log10(1 / MSE_Loss)
    print('PSNR = {}dB'.format(psnr))
    print('output image saved to ' +  output_filename)
    MSE=MSE+MSE_Loss
    idx=idx+1
        
MSE_avg=MSE/idx
psnr_avg=10 * log10(1 / MSE_avg)
print('Average PSNR = {}dB'.format(psnr_avg))
