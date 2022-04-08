clc; clear;
rng default

Train_filedir='D:\Dennis_Estrada_Image_Enhancement\image_datasets\Train_HR';
Valid_filedir='D:\Dennis_Estrada_Image_Enhancement\image_datasets\Valid_HR';
DIR_train=dir(fullfile(Train_filedir,'*.png'));
DIR_valid=dir(fullfile(Valid_filedir,'*.png'));

savedir='D:\Dennis_Estrada_Image_Enhancement\MF_Turbulence\images_3';
Input_folder=[savedir,'\train\A'];
Input_valid_folder=[savedir,'\test\A'];
Target_folder=[savedir,'\train\B'];
Target_valid_folder=[savedir,'\test\B'];
mkdir(savedir)
mkdir(Input_folder)
mkdir(Input_valid_folder)
mkdir(Target_folder)
mkdir(Target_valid_folder)

n_files_train=3350;
n_files_valid=200;
N_scan=9;
N_per_train=3;
N_per_valid=1;

CS=512;
CS_local=64;
pad=CS_local;

Noise=1; %% 0 No Noise, 1 Shot/Poisson Noise
Gray=1;

param.D=0.01; %Aperture diameter in meters
param.d=0.05; %Focal length in meters
param.pixel_sep=2.5*10^-6; %pixel seperation of sensor
param.lambda=550*10^-9; %Wavelength
param.N=15; %Size of PSF filter
param.L = 5; %Distance from target
%param.Cn2 = 1e-10; %refractivity structure index parameter
param.N_psf_gen=64; %Number of PSF to generate

rand_opts.noise_shot=0;
rand_opts.noise_scale_min=10^(7);
rand_opts.noise_scale_max=10^(9);

c=0;
index=randperm(n_files_train*N_per_train);
for i=1:n_files_train
    fid=DIR_train(i).name;
    img=im2double(imread(fullfile(Train_filedir,fid)));
    img=imresize(img,[1024,1024]);
    W=size(img,1);
    H=size(img,2);
    if Gray == 1
        img=rgb2gray(img);
    end
    
    for j=1:N_per_train
        noise_scale=rand_opts.noise_scale_min+(rand(1)*(rand_opts.noise_scale_max-rand_opts.noise_scale_min));
        param.Cn2 =(1+rand(1)*10)*10^-10;   %-  index of refraction structure

        c=c+1;
        input_scan_folder=fullfile(Input_folder,['scan_',num2str(index(c),'%05i')]);
        target_scan_folder=fullfile(Target_folder,['scan_',num2str(index(c),'%05i')]);
        mkdir(input_scan_folder)
        mkdir(target_scan_folder)
        
        X=1+round(rand(1)*(H-CS));
        Y=1+round(rand(1)*(W-CS));
        target=imcrop(img,[X,Y,CS-1,CS-1]);
        
        for h=1:N_scan
            degraded = sim_turbulence(target,param);
            
            if Noise == 1
                degraded=noise_shot(degraded,noise_scale,rand_opts);
            end
            
            degraded_stack(:,:,1,h)=degraded;
        end

        degraded_stack(:,:,:,N_scan+1)=target;

        [~, global_Tvec]=Register_DFT(degraded_stack,N_scan+1,1);
        align_global=UpSample_Shift(degraded_stack,global_Tvec,1,1);

        X=1+round(rand(1)*(CS-CS_local-pad));
        Y=1+round(rand(1)*(CS-CS_local-pad));
        
        for h=1:N_scan+1
            img_crop=align_global(:,:,:,h);
            img_crop=imcrop(img_crop,[X,Y,CS_local+pad-1,CS_local+pad-1]);
            align_crop(:,:,:,h)=img_crop;
        end

        [~, local_Tvec]=Register_DFT(align_crop,N_scan+1,1);
        align_crop_local=UpSample_Shift(align_crop,local_Tvec,1,1);

        win=centerCropWindow2d([size(align_crop_local,1),size(align_crop_local,2)],[CS_local,CS_local]);
        target=align_crop_local(:,:,:,N_scan+1);
        target=imcrop(target,win);

        for h=1:N_scan
            degraded=align_crop_local(:,:,:,h);
            degraded=imcrop(degraded,win);
            
            imwrite(target,fullfile(target_scan_folder,['image_',num2str(h,'%05i'),'.png']))
            imwrite(degraded,fullfile(input_scan_folder,['image_',num2str(h,'%05i'),'.png']))
        end

    end
end

c=0;
index=randperm(n_files_valid*N_per_valid);
for i=1:n_files_valid
    fid=DIR_valid(i).name;
    img=im2double(imread(fullfile(Valid_filedir,fid)));
    img=imresize(img,[1024,1024]);
    W=size(img,1);
    H=size(img,2);
    
    if Gray == 1
        img=rgb2gray(img);
    end
    
    for j=1:N_per_valid
        noise_scale=rand_opts.noise_scale_min+(rand(1)*(rand_opts.noise_scale_max-rand_opts.noise_scale_min));
        param.Cn2 =(1+rand(1)*10)*10^-10;   %-  index of refraction structure
        
        c=c+1;
        input_scan_folder=fullfile(Input_valid_folder,['scan_',num2str(index(c),'%05i')]);
        target_scan_folder=fullfile(Target_valid_folder,['scan_',num2str(index(c),'%05i')]);
        mkdir(input_scan_folder)
        mkdir(target_scan_folder)
        
        X=1+round(rand(1)*(H-CS));
        Y=1+round(rand(1)*(W-CS));
        target=imcrop(img,[X,Y,CS-1,CS-1]);
        
        for h=1:N_scan
            degraded = sim_turbulence(target,param);
            
            if Noise == 1
                degraded=noise_shot(degraded,noise_scale,rand_opts);
            end
            
            degraded_stack(:,:,1,h)=degraded;
        end

        degraded_stack(:,:,:,N_scan+1)=target;

        [~, global_Tvec]=Register_DFT(degraded_stack,N_scan+1,1);
        align_global=UpSample_Shift(degraded_stack,global_Tvec,1,1);

        X=1+round(rand(1)*(CS-CS_local-pad));
        Y=1+round(rand(1)*(CS-CS_local-pad));
        
        for h=1:N_scan+1
            img_crop=align_global(:,:,:,h);
            img_crop=imcrop(img_crop,[X,Y,CS_local+pad-1,CS_local+pad-1]);
            align_crop(:,:,:,h)=img_crop;
        end

        [~, local_Tvec]=Register_DFT(align_crop,N_scan+1,1);
        align_crop_local=UpSample_Shift(align_crop,local_Tvec,1,1);

        win=centerCropWindow2d([size(align_crop_local,1),size(align_crop_local,2)],[CS_local,CS_local]);
        target=align_crop_local(:,:,:,N_scan+1);
        target=imcrop(target,win);

        for h=1:N_scan
            degraded=align_crop_local(:,:,:,h);
            degraded=imcrop(degraded,win);
            
            imwrite(target,fullfile(target_scan_folder,['image_',num2str(h,'%05i'),'.png']))
            imwrite(degraded,fullfile(input_scan_folder,['image_',num2str(h,'%05i'),'.png']))
        end
    end
end

