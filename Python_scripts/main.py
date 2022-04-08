from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
import torchvision
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from datasets import *
from networks import *
from losses import *
from fusion import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enables test during training')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=4)
parser.add_argument('--dataroot', default="images_3/train", type=str, help='path to dataset')
parser.add_argument('--testroot', default="images_3/test", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=500)
parser.add_argument('--testsize', type=int, help='number of testing data', default=10)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=10, help='test batch size')
parser.add_argument('--save_iter', type=int, default=50, help='the interval iterations for saving models')
parser.add_argument('--cdim', type=int, default=1, help='the channel-size  of the input image to network')
parser.add_argument('--input_height', type=int, default=64, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
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
parser.add_argument("--loss_2_weight", type=float, default=1, help="loss 2 weight")
parser.add_argument("--loss_dis", type=int, default=2, help="loss type")
parser.add_argument("--N_input_images", type=int, default=9, help="upsample scale")
parser.add_argument("--num_dis_updates", type=int, default=5, help="Number of Discriminator updates")
parser.add_argument("--adv_loss_weight", type=float, default=0.01, help="loss type")


fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
Epoch_Num = []
Avg_Train_loss = []
Avg_Valid_loss = []

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    ngpu = int(opt.ngpu)

    model_name='Weight_Generator'
    wtnet = weights_generator(num_channels=opt.cdim, base_filter=32,num_input_images=opt.N_input_images)
    model_dis_name='Image_Discrminator'
    srdis= discriminator(num_channels=opt.cdim, base_filter=32, input_height=opt.input_height, input_width=opt.input_width)


    if os.path.isfile(opt.pretrained_model):
        print("=> loading model '{}'".format(opt.pretrained_model))
        weights = torch.load(opt.pretrained_model)
        pretrained_dict = weights['model'].state_dict()
        model_dict = wtnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        wtnet.load_state_dict(model_dict)
    else:
        print("=> no generator model found at '{}'".format(opt.pretrained_model))

    if os.path.isfile(opt.pretrained_model_dis):
        weights_dis = torch.load(opt.pretrained_model_dis)
        pretrained_dict_dis = weights_dis['model'].state_dict()
        model_dict_dis = srdis.state_dict()
        pretrained_dict_dis = {k: v for k, v in pretrained_dict_dis.items() if k in model_dict_dis}
        model_dict_dis.update(pretrained_dict_dis) 
        srdis.load_state_dict(model_dict_dis)
    else:
        print("=> no discriminator model found at '{}'".format(opt.pretrained_model_dis))

    if opt.loss == 1:
        loss_name='L1'
    elif opt.loss == 2:
        loss_name='MSE'
    elif opt.loss == 3:
        loss_name='correntropy_025'
        corr_sigma=0.25
    elif opt.loss == 4:
        loss_name='correntropy_050'
        corr_sigma=0.5
    elif opt.loss == 5:
        loss_name='correntropy_075'
        corr_sigma=0.75
    elif opt.loss == 6:
        loss_name='VGG16'
        loss_VGG=loss_VGG16(feature_layers=[1])
        loss_VGG.eval()
    elif opt.loss == 7:
        loss_name='VGG19'
        loss_VGG=loss_VGG19(feature_layers=[1])
        loss_VGG.eval()
    elif opt.loss == 8:
        loss_name='FFT'
    elif opt.loss == 9:
        loss_name='Wavelet'
        wavelet_dec = WaveletTransform(scale=opt.upscale, dec=True)
    elif opt.loss -- 10:
        loss_name='FFT_VGG16'
        loss_VGG=loss_VGG16(feature_layers=[1])
        loss_VGG.eval()
        
        
    if opt.loss_2==1:
        loss_name=loss_name+'_L1'
    elif opt.loss_2 == 2:
        loss_name=loss_name+'_MSE'
    elif opt.loss_2 == 3:
        loss_name=loss_name+'_correntropy_025'
        corr_sigma_2=0.25
    elif opt.loss_2 == 4:
        loss_name=loss_name+'_correntropy_050'
        corr_sigma_2=0.5
    elif opt.loss_2 == 5:
        loss_name=loss_name+'_correntropy_075'
        corr_sigma_2=0.75
        

    if opt.loss_dis == 1:
        disloss_name='GAN'
        gan_loss=GANLoss(use_l1=False)
    elif opt.loss_dis == 2:
        disloss_name='GANLS'
        gan_loss=GANLoss(use_l1=True)
    elif opt.loss_dis == 3:
        disloss_name='WGANGP'
        gan_loss=GANLoss()

    if opt.cuda:
        wtnet = wtnet.cuda()
        srdis=srdis.cuda()
        gan_loss=gan_loss.cuda()
        if opt.loss == 6 or opt.loss == 7 or opt.loss == 10:
            loss_VGG=loss_VGG.cuda()
    
    optimizer_wt = optim.Adam(wtnet.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0005)
    model_savename=model_name+'_'+loss_name
    
    optimizer_dis = optim.Adam(srdis.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0005)
    model_savename+=model_dis_name+'_'+disloss_name


    train_list_A, train_list_B = load_list(opt.dataroot, opt.trainsize)
    test_list_A, test_list_B = load_list(opt.testroot, opt.testsize)

    print(len(train_list_A))
    train_set = ImageDatasetFromFile(train_list_A, train_list_B, 
              input_height=opt.input_height, input_width=opt.input_width,N_input_images=opt.N_input_images)    
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    
    
    test_set = ImageDatasetFromFile(test_list_A,test_list_B, 
                  input_height=opt.input_height, input_width=opt.input_width,N_input_images=opt.N_input_images)    
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
                                         
    total_iters=opt.nEpochs*len(train_data_loader)
    n_iters=0    
    start_time = time.time()
    wtnet.train()
    srdis.train()

    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs):  
        losses=0
        for iteration, batch in enumerate(train_data_loader, 0):
            n_iters=n_iters+1 
            #--------------train------------
            
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            wtnet.eval()
            srdis.train()
            
            predict_weights, predict = forward_parallel(wtnet, input, opt.ngpu)
            
            
            for i in range(opt.num_dis_updates):
                optimizer_dis.zero_grad()
                
                fake_d =forward_parallel(srdis, predict, opt.ngpu)
                real_d =forward_parallel(srdis, target, opt.ngpu)
                    
                if opt.loss_dis == 1 or opt.loss_dis == 2:
                    real_tensor = torch.FloatTensor(real_d.size()).fill_(1.0)
                    real_label_var = Variable(real_tensor, requires_grad=False)

                    fake_tensor = torch.FloatTensor(fake_d.size()).fill_(0.0)
                    fake_label_var = Variable(fake_tensor, requires_grad=False)

                    if opt.cuda:
                        real_label_var=real_label_var.cuda()
                        fake_label_var=fake_label_var.cuda()
                            
                    d_loss_real=forward_parallel_loss(gan_loss, real_d, real_label_var, opt.ngpu)
                    d_loss_fake=forward_parallel_loss(gan_loss, fake_d, fake_label_var, opt.ngpu)
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                elif opt.loss_dis == 3:
                    alpha = torch.rand(1, 1)
                    alpha = alpha.expand(target.size())
                        
                    if opt.cuda:
                        alpha = alpha.cuda()

                    interpolates = alpha * target + ((1 - alpha) * predict)

                    interpolates = Variable(interpolates, requires_grad=True)
                    if opt.cuda:
                        interpolates = interpolates.cuda()
                            
                    disc_interpolates = forward_parallel(srdis, interpolates, opt.ngpu)

                    grad_out=torch.ones(disc_interpolates.size())
                    if opt.cuda:
                        grad_out=grad_out.cuda()
                            
                    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                                        grad_outputs=grad_out,create_graph=True, retain_graph=True,
                                                        only_inputs=True)[0]
                        
                    LAMBDA = 10
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA


                    d_loss =  fake_d.mean() - real_d.mean() + gradient_penalty
                    wass_loss = real_d.mean()-fake_d.mean()
                        
                d_loss.backward(retain_graph=True)
                optimizer_dis.step()

            wtnet.train()
            srdis.eval()

            optimizer_wt.zero_grad()
            predict_weights, predict = forward_parallel(wtnet, input, opt.ngpu)
            
            
            if opt.loss == 1:
                loss_img = loss_L1(predict, target, opt.mse_avg)
            elif opt.loss == 2:
                loss_img = loss_MSE(predict, target, opt.mse_avg)
            elif opt.loss == 3 or opt.loss == 4 or opt.loss == 5:
                loss_img = loss_correntropy(predict,target,sigma=corr_sigma)
            elif opt.loss == 6 or opt.loss == 7:
                loss_img = forward_parallel_loss(loss_VGG, predict,target, ngpu)
            elif opt.loss == 8:
                loss_img=FFT_loss(predict,target)
                loss_img=loss_img.mul(0.01)
            elif opt.loss == 9:
                target_wavelet=wavelet_dec(target)
                if not opt.model == 8:
                    predict_wavelet=wavelet_dec(predict)
                else:
                    predict_wavelet=predict
            
                wavelets_lr = predict_wavelet[:,0:3,:,:]
                wavelets_sr = predict_wavelet[:,3:,:,:]
            
                loss_lr = loss_MSE(target_wavelet[:,0:3,:,:], wavelets_lr, opt.mse_avg)
                loss_sr = loss_MSE(target_wavelet[:,3:,:,:], wavelets_sr, opt.mse_avg)
            
                loss_textures = loss_Textures(target_wavelet[:,3:,:,:], wavelets_sr)
                loss_img = loss_sr.mul(0.99) + loss_lr.mul(0.01) + loss_textures.mul(1)
            elif opt.loss == 10:
                loss_img_VGG = forward_parallel_loss(loss_VGG, predict,target, ngpu)
                loss_img_FFT=FFT_loss(predict,target)
                loss_img=loss_img_VGG.mul(0.1)+loss_img_FFT
                
            if opt.loss_2 == 1:
                loss_img_2 = loss_L1(predict, target, opt.mse_avg)
                loss_img=loss_img+loss_img_2.mul(opt.loss_2_weight)
            elif opt.loss_2 == 2:
                loss_img_2 = loss_MSE(predict, target, opt.mse_avg)
                loss_img=loss_img+loss_img_2.mul(opt.loss_2_weight)
            elif opt.loss_2 == 3 or opt.loss_2 == 4 or opt.loss_2 == 5:
                loss_img_2 = loss_correntropy(predict,target,sigma=corr_sigma_2)
                loss_img=loss_img+loss_img_2.mul(opt.loss_2_weight)


            fake_d =forward_parallel(srdis, predict, opt.ngpu)
            if opt.loss_dis == 1 or opt.loss_dis == 2:
                adv_loss=forward_parallel_loss(gan_loss, fake_d, real_label_var, opt.ngpu)
            elif opt.loss_dis == 3:
                adv_loss=-fake_d.mean()

            loss = loss_img + (opt.adv_loss_weight*adv_loss)
            loss.backward()                       
            optimizer_wt.step()
            losses+=loss.item()
            
            cur_time=(time.time()-start_time)/60
            time_iter=cur_time/n_iters
            est_time=cur_time+(time_iter*(total_iters-n_iters))
            Hc=np.floor(cur_time/60)
            Mc=np.floor(cur_time-Hc*60)
            Sc=((cur_time-Hc*60)-Mc)*60
            He=np.floor(est_time/60)
            Me=np.floor(est_time-He*60)
            Se=((est_time-He*60)-Me)*60
            
            info = "===> Epoch:[{}/{}]({}/{}), ".format(epoch+1, opt.nEpochs, iteration+1, len(train_data_loader))
            info += "Total Time: [{}:{}:{:2.2f}]/[{}:{}:{:2.2f}], ".format(Hc.astype(int),Mc.astype(int),Sc,
                                                                       He.astype(int),Me.astype(int),Se)
            info += "G-Loss: {:.4f}".format(loss.item())            
            info += ", D-Loss {:.4f}".format(d_loss.item())
            info += ", W-Loss: {:.4f}".format(wass_loss.item())
            print(info)
            
        #--------------test-------------
        if opt.test:
            wtnet.eval()
            avg_psnr = 0
            for titer, batch in enumerate(test_data_loader,0):
                input, target = Variable(batch[0]), Variable(batch[1])
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()
                predict_weights, prediction = forward_parallel(wtnet, input, opt.ngpu)
                
            
                mse = criterion_MSE(prediction, target)
                psnr = 10 * log10(1 / (mse.item()) )
                avg_psnr += psnr

                save_images_merged(input, predict_weights, prediction, target,"Epoch_{:03d}_Iter_{:06d}_{:02d}.jpg".format(epoch+1, iteration+1, titer+1),path=model_savename,)
                 
            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))
            wtnet.train()
            plot_progress(losses/len(train_data_loader),avg_psnr / len(test_data_loader),epoch,model_savename)
            save_valid_losses(Avg_Valid_loss,model_savename)
            
        save_train_losses(Avg_Train_loss,model_savename)
            
        if (epoch+1)%opt.save_iter == 0:
            save_checkpoint(wtnet, epoch+1, model_savename)

    

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)

def forward_parallel_loss(net, input, target, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, (input, target), range(ngpu))
    else:
        return net(input,target)
            
def save_checkpoint(model, epoch, prefix=""):
    modelroot="model"
    modelfolder=modelroot+"/" + prefix
    model_out_path =  modelfolder+"/epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def save_images_merged(inputs,weights,predicts,targets, name, path):
    resultspath=opt.outf+'/'+path
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
    input = inputs.cpu()
    weight = weights.cpu()
    predict = predicts.cpu()
    target = targets.cpu()
    ins= input.data.numpy().astype(np.float32)
    weis= weight.data.numpy().astype(np.float32)
    pres= predict.data.numpy().astype(np.float32)
    tars= target.data.numpy().astype(np.float32)
    ins=ins.transpose(0,2,3,4,1)
    weis=weis.transpose(0,2,3,4,1)
    pres=pres.transpose(0,2,3,1)
    tars=tars.transpose(0,2,3,1)
    
    imsave_stack(ins,weis,pres,tars, os.path.join(resultspath, name))

def imsave_stack(inputs, weights, predicts, targets, path):
    img = merge_stack(inputs, weights, predicts, targets)
    image = Image.fromarray(img.astype(np.uint8))
    return image.save(path)

def merge_stack(inputs, weights, predicts, targets):
    N_scans=np.shape(inputs)[0]
    N_images=np.shape(inputs)[1]
    H=np.shape(inputs)[2]
    W=np.shape(inputs)[3]
    img = np.zeros((H * N_scans, W * (N_images*2+2), 3))
    line1 = np.zeros((H, W * (N_images), 3))
    line2 = np.zeros((H, W * (N_images), 3))
    for i in range(N_scans):
        for j in range(N_images):
            img_in = inputs[i,j,:,:,:]
            weight = weights[i,j,:,:,:]
            line1[:,j*W:j*W+W,:] =img_in
            line2[:,j*W:j*W+W,:] =weight
        img[i*H:i*H+H,0:W*N_images,:]=line1
        img[i*H:i*H+H,W*N_images:2*W*N_images,:]=line2
        img[i*H:i*H+H,2*W*N_images:2*W*N_images+W,:]=predicts[i,:,:,:]
        img[i*H:i*H+H,2*W*N_images+W:2*W*N_images+2*W,:]=targets[i,:,:,:]
    img = img*255
    
    return img


def plot_progress(Train_loss,Valid_loss,Epoch_Number,path):
    resultspath=opt.outf+'/'+path
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
        
    
    Epoch_Num.append(Epoch_Number)
    Avg_Train_loss.append(Train_loss)
    Avg_Valid_loss.append(Valid_loss)

    ax1.plot(Epoch_Num,Avg_Train_loss)
    ax1.set_xlabel('Epoch Number')
    ax1.set_title('Average Training Loss') 
    
    ax2.plot(Epoch_Num,Avg_Valid_loss)
    ax2.set_xlabel('Epoch')
    ax2.set_title('Average Validation PSNR')
    
    plt.draw()
    plt.pause(1)
    plt.show
    
    plt.savefig(os.path.join(resultspath,path+'_Train_Valid_stats.png'))
    
    return

def save_train_losses(train_losses,path):
    resultspath=opt.outf+'/'+path
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
        
    filename = os.path.join(resultspath,path+'_Train_losses.mat')
    
    sio.savemat(filename,{'train_losses':train_losses})
    
    return

def save_valid_losses(valid_losses,path):
    resultspath=opt.outf+'/'+path
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
        
    filename = os.path.join(resultspath,path+'_Valid_losses.mat')
    
    sio.savemat(filename,{'valid_losses':valid_losses})
    
    return

if __name__ == "__main__":
    main() 



    
