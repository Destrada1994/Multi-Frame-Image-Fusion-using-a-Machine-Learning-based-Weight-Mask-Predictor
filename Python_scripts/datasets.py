import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import isfile, join, basename
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
from scipy import signal, ndimage
import math
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])  

def load_list(path,datasize):
    if path is None:
          return None, None
    print("Loading files from %s" % path)
    path_A=join(path,'A')
    path_B=join(path,'B')
    idx=0
    list_A=[]
    for scan in listdir(path_A):
        scan_id=basename(scan)
        if idx < datasize:
            scan_path=join(path_A,scan_id)
            list_A.append(scan_path)
            idx=idx+1
        else:
            break
            
    idx=0
    list_B=[]
    for scan in listdir(path_B):
        scan_id=basename(scan)
        if idx < datasize:
            scan_path=join(path_B,scan_id)
            list_B.append(scan_path)
            idx=idx+1
        else:
            break

    return list_A, list_B

def load_single_list(path,datasize):
    if path is None:
          return None, None
    print("Loading files from %s" % path)
    idx=0
    list_A=[]
    for scan in listdir(path):
        scan_id=basename(scan)
        if idx < datasize:
            scan_path=join(path,scan_id)
            list_A.append(scan_path)
            idx=idx+1
        else:
            break

    return list_A

def load_video_image(file_path_A, file_path_B, input_height=128, input_width=None,N_input_images=4):
    
    if input_width is None:
      input_width = input_height

    output_width=input_width
    output_height=input_height

    imgs_A=[]
    An=0
    for image_name in listdir(file_path_A):
        image_path=join(file_path_A,image_name)
        if isfile(image_path):
            if An < N_input_images:
                img_A=Image.open(image_path)
                #img_A = img_A.convert('RGB')
                width, height = img_A.size   # Get dimensions
        
                #Center Crop
                if width > input_width:
                    left = math.floor((width - input_width)/2)
                    right = left+input_width
                else:
                    left=0
                    right=width
                if height > input_height:
                    top = math.floor((height - input_height)/2)
                    bottom = top+input_height
                else:
                    top=0;
                    bottom=height;
        
                img_A = img_A.crop((left, top, right, bottom)) #Crop Image 1
                img_A=np.array(img_A)
                
                imgs_A.append(img_A)
            else:
                break

    imgs_B=[]
    for image_name in listdir(file_path_B):
        image_path=join(file_path_B,image_name)
        if isfile(image_path):
            img_B=Image.open(image_path)
            #img_B = img_B.convert('RGB')
            width, height = img_B.size   # Get dimensions
    
            #Center Crop
            if width > input_width:
                left = math.floor((width - input_width)/2)
                right = left+input_width
            else:
                left=0
                right=width
            if height > input_height:
                top = math.floor((height - input_height)/2)
                bottom = top+input_height
            else:
                top=0;
                bottom=height;
    
            img_B = img_B.crop((left, top, right, bottom)) #Crop Image 1
            break
    imgs_A=np.array(imgs_A)
    imgs_A=np.expand_dims(imgs_A,axis=-1)
    imgs_A=imgs_A.transpose((3,0,1,2))
    imgs_A=imgs_A.astype('float32')/255
    imgs_B=np.array(img_B)
    imgs_B=np.expand_dims(imgs_B,axis=-1)
    imgs_B=imgs_B.transpose((2,0,1))
    imgs_B=imgs_B.astype('float32')/255
    return imgs_A, imgs_B

def load_valid_image(file_path_A, file_path_B, N_input_images=4):
    
    imgs_A=[]
    An=0
    for image_name in listdir(file_path_A):
        image_path=join(file_path_A,image_name)
        if isfile(image_path):
            if An < N_input_images:
                img_A=Image.open(image_path)
                #img_A = img_A.convert('RGB')
                input_width, input_height = img_A.size
                img_A=np.array(img_A)

                imgs_A.append(img_A)
            else:
                break
    
    imgs_B=[]
    for image_name in listdir(file_path_B):
        image_path=join(file_path_B,image_name)
        if isfile(image_path):
            img_B=Image.open(image_path)
            #img_B = img_B.convert('RGB')
            break
        
    imgs_A=np.array(imgs_A)
    imgs_A=np.expand_dims(imgs_A,axis=-1)
    imgs_A=imgs_A.transpose((3,0,1,2))
    imgs_A=imgs_A.astype('float32')/255
    imgs_B=np.array(img_B)
    imgs_B=np.expand_dims(imgs_B,axis=-1)    
    imgs_B=imgs_B.transpose((2,0,1))
    imgs_B=imgs_B.astype('float32')/255
    return imgs_A, imgs_B

def load_single_valid_image(file_path_A, N_input_images=4):
    
    imgs_A=[]
    An=0
    for image_name in listdir(file_path_A):
        image_path=join(file_path_A,image_name)
        if isfile(image_path):
            if An < N_input_images:
                img_A=Image.open(image_path)
                #img_A = img_A.convert('RGB')
                input_width, input_height = img_A.size
                img_A=np.array(img_A)

                imgs_A.append(img_A)
            else:
                break
        
    imgs_A=np.array(imgs_A)
    imgs_A=np.expand_dims(imgs_A,axis=-1)
    imgs_A=imgs_A.transpose((3,0,1,2))
    imgs_A=imgs_A.astype('float32')/255
    return imgs_A

class ImageDatasetFromFile(data.Dataset):
    def __init__(self, scan_list_A, scan_list_B, input_height=128,
                 input_width=None,N_input_images=4):
        
        super(ImageDatasetFromFile, self).__init__()
        self.N_input_images=N_input_images  
        self.scan_A_filenames = scan_list_A
        self.scan_B_filenames = scan_list_B
        self.input_height = input_height
        if input_width ==None:
            self.input_width = input_height
        else:
            self.input_width = input_width


        #self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        lr, hr = load_video_image(self.scan_A_filenames[index],
                                  self.scan_B_filenames[index], 
                                  self.input_height, self.input_width,self.N_input_images)

        input = torch.from_numpy(lr)
        target = torch.from_numpy(hr)
        return input, target

    def __len__(self):
        return len(self.scan_A_filenames)



