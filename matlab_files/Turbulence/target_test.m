clc; clear;
rng default

Train_filedir='E:\matlab_code\Turbulance_MF\SF_Turbulance\images\Target_usaf';
DIR_train=dir(fullfile(Train_filedir,'*.png'));


n_files_train=1;
n_files_valid=1;

N_per_train=5;
N_per_valid=0;

CS=512;
c=0;
for i=1:n_files_train
    fid=DIR_train(i).name;
    img=double(imread(fullfile(Train_filedir,fid)))/255;
    %img=imresize(img,[1024,1024]);
    %img=(img-min(img(:)))/(max(img(:))-min(img(:)));
    img(img<0.5)=0;
    img(img>0.5)=1;
    
    figure(1)
    imshow(img) 
    
    
    W=size(img,1);
    H=size(img,2);
    
    degraded=img;
    
    r_rand=(0.55);
    g_rand=(0.12);
    b_rand=(0.11);
    
    degraded(:,:,1)=degraded(:,:,1)-r_rand;
    degraded(:,:,2)=degraded(:,:,2)-g_rand;
    degraded(:,:,3)=degraded(:,:,3)-b_rand;
    
    degraded(degraded<0)=0;
    
    noise_red=rand(size(degraded,1),size(degraded,2))*0.01;
    noise_green=0.09+rand(size(degraded,1),size(degraded,2))*0.01;
    noise_blue=0.09+rand(size(degraded,1),size(degraded,2))*0.01;

    noise=cat(3,noise_red,noise_green,noise_blue);
    
    degraded=degraded+noise;
    
    figure(2)
    imshow(degraded)
end
