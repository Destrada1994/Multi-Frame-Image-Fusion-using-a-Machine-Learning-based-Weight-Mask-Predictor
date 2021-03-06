clc; clear;
rng default

Train_filedir='E:\matlab_code\Turbulance_MF\SF_Turbulance\images\Target_usaf';
Valid_filedir='E:\matlab_code\Turbulance_MF\SF_Turbulance\images\Target_usaf';
DIR_train=dir(fullfile(Train_filedir,'*.png'));
DIR_valid=dir(fullfile(Valid_filedir,'*.png'));

savedir='E:\matlab_code\Turbulance_MF\SF_Turbulance\images_4';
Input_folder=[savedir,'\Input'];
Input_valid_folder=[savedir,'\Input_valid'];
Target_folder=[savedir,'\Target'];
Target_valid_folder=[savedir,'\Target_valid'];
mkdir(savedir)
mkdir(Input_folder)
mkdir(Input_valid_folder)
mkdir(Target_folder)
mkdir(Target_valid_folder)

n_files_train=1;
n_files_valid=1;

N_per_train=1000;
N_per_valid=100;

CS=128;

%     params  -  object containing all parameters
%     params.t_params  -  object containing turbulence parameters
params.t_params.d = 0.01;     %-  focal length (m) {1.2}
params.t_params.D = 0.005;     %-  aperture diameter size (m) {0.2034}
params.t_params.lambda = 0.525e-6;%-  wavelength (m) {0.525e-6}
params.t_params.L = 5;     %-  propagation length (m) {7000}
%params.t_params.Cn2 =5e-16;   %-  index of refraction structure
%                                 parameter {5e-16} (yes, it's that 
%                                 small! 5e-17 to 5e-15 is a good 
%                                 basic range.)
params.t_params.k = 2*pi/params.t_params.lambda;     %-  wave number (rad/m) 
%                                 {2*pi/params.t_params.lambda}
%params.s_params  %-  object containing sampling parameters
params.s_params.rowsW = 64;  %-  number of rows in phase (pixels) {64}
params.s_params.colsW = 64; %-  number of rows in phase (pixels) {64}
params.s_params.fftK = 2;  %-  upsampling ratio {2}
params.s_params.K = 16;     %-  number of PSFs used per row {16}
params.s_params.T = params.s_params.K^2;    %-  PSFs in total image {params.s_params.K^2}
opt.frames=1;


c=0;
index=randperm(n_files_train*N_per_train);
for i=1:n_files_train
    fid=DIR_train(i).name;
    img=double(imread(fullfile(Train_filedir,fid)))/255;
    %img=imresize(img,[1024,1024]);
    %img=(img-min(img(:)))/(max(img(:))-min(img(:)));
    
    W=size(img,1);
    H=size(img,2);
    
    for j=1:N_per_train
        c=c+1;
        X=1+round(rand(1)*(H-CS));
        Y=1+round(rand(1)*(W-CS));
        
        target=imcrop(img,[X,Y,CS-1,CS-1]);
        
        params.t_params.Cn2 =(1+rand(1)*5)*10^-10;   %-  index of refraction structure
        
        degraded=target;
        
        degraded = sim_fun(degraded,params,opt);
        
        %add blue tint
        r_rand=(0.55+(rand(1)*0.01));
        g_rand=(0.12+(rand(1)*0.01));
        b_rand=(0.11+(rand(1)*0.01));
        
        degraded(:,:,1)=degraded(:,:,1)-r_rand;
        degraded(:,:,2)=degraded(:,:,2)-g_rand;
        degraded(:,:,3)=degraded(:,:,3)-b_rand;
        
        target(:,:,1)=target(:,:,1)-r_rand;
        target(:,:,2)=target(:,:,2)-g_rand;
        target(:,:,3)=target(:,:,3)-b_rand;
        
        degraded(degraded<0)=0;
        target(target<0)=0;
        
        
        noise_red=rand(size(degraded,1),size(degraded,2))*0.01;
        noise_green=0.09+rand(size(degraded,1),size(degraded,2))*0.01;
        noise_blue=0.09+rand(size(degraded,1),size(degraded,2))*0.01;
        
        noise=cat(3,noise_red,noise_green,noise_blue);
        
        degraded=degraded+noise;
        
        imwrite(target,fullfile(Target_folder,['image_',num2str(index(c),'%05i'),'.png']))
        imwrite(degraded,fullfile(Input_folder,['image_',num2str(index(c),'%05i'),'.png']))
    end
end

c=0;
index=randperm(n_files_valid*N_per_valid);
for i=1:n_files_valid
    fid=DIR_valid(i).name;
    img=double(imread(fullfile(Valid_filedir,fid)))/255;
    %img=imresize(img,[1024,1024]);
    %img=(img-min(img(:)))/(max(img(:))-min(img(:)));
    
    W=size(img,1);
    H=size(img,2);
    
    for j=1:N_per_valid
        c=c+1;
        X=1+round(rand(1)*(H-CS));
        Y=1+round(rand(1)*(W-CS));
        
        target=imcrop(img,[X,Y,CS-1,CS-1]);
        
        degraded = sim_fun(target,params,opt);
        
        %add blue tint
        r_rand=(0.55+(rand(1)*0.01));
        g_rand=(0.12+(rand(1)*0.01));
        b_rand=(0.11+(rand(1)*0.01));
        
        degraded(:,:,1)=degraded(:,:,1)-r_rand;
        degraded(:,:,2)=degraded(:,:,2)-g_rand;
        degraded(:,:,3)=degraded(:,:,3)-b_rand;
        
        target(:,:,1)=target(:,:,1)-r_rand;
        target(:,:,2)=target(:,:,2)-g_rand;
        target(:,:,3)=target(:,:,3)-b_rand;
        
        degraded(degraded<0)=0;
        target(target<0)=0;
        
        
        noise_red=rand(size(degraded,1),size(degraded,2))*0.01;
        noise_green=0.09+rand(size(degraded,1),size(degraded,2))*0.01;
        noise_blue=0.09+rand(size(degraded,1),size(degraded,2))*0.01;
        
        noise=cat(3,noise_red,noise_green,noise_blue);
        
        degraded=degraded+noise;
        
        imwrite(target,fullfile(Target_valid_folder,['image_',num2str(index(c),'%05i'),'.png']))
        imwrite(degraded,fullfile(Input_valid_folder,['image_',num2str(index(c),'%05i'),'.png']))
    end
end