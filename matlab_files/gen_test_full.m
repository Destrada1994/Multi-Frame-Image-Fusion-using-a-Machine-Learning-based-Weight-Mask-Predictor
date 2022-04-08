clc; clear;
rng default

Valid_filedir='D:\Dennis_Estrada_Image_Enhancement\image_datasets\Valid_HR';
DIR_valid=dir(fullfile(Valid_filedir,'*.png'));

savedir='D:\Dennis_Estrada_Image_Enhancement\MF_Turbulence\images_3\full_Low';
Input_valid_folder=[savedir,'\A'];
Target_valid_folder=[savedir,'\B'];
mkdir(savedir)
mkdir(Input_valid_folder)
mkdir(Target_valid_folder)

n_files_valid=200;
N_scan=9;
N_per_valid=1;

CS=512;
CS_local=256;
pad=CS_local;

Noise=1; %% 0 No Noise, 1 Shot/Poisson Noise
Gray=1;

param.D=0.01; %Aperture diameter in meters
param.d=0.05; %Focal length in meters
param.pixel_sep=2.5*10^-6; %pixel seperation of sensor
param.lambda=550*10^-9; %Wavelength
param.N=15; %Size of PSF filter
param.L = 5; %Distance from target
param.Cn2 = 5e-10; %refractivity structure index parameter
param.N_psf_gen=64; %Number of PSF to generate

rand_opts.noise_shot=0;
rand_opts.noise_scale_min=10^(6);
rand_opts.noise_scale_max=10^(9);
noise_scale=10^(8.5);

c=0;
index=randperm(n_files_valid*N_per_valid);
for i=1:n_files_valid
    fid=DIR_valid(i).name;
    img=im2double(imread(fullfile(Valid_filedir,fid)));
    img=imresize(img,[1024,1024]);
    W=size(img,1);
    H=size(img,2);
    CS=min([W,H]);
    if Gray == 1
        img=rgb2gray(img);
    end
    
    for j=1:N_per_valid
%         noise_scale=rand_opts.noise_scale_min+(rand(1)*(rand_opts.noise_scale_max-rand_opts.noise_scale_min));
%         param.Cn2 =(1+rand(1)*5)*10^-10;   %-  index of refraction structure
%         
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

        for h=1:N_scan
            degraded=degraded_stack(:,:,:,h);
            
            imwrite(target,fullfile(target_scan_folder,['image_',num2str(h,'%05i'),'.png']))
            imwrite(degraded,fullfile(input_scan_folder,['image_',num2str(h,'%05i'),'.png']))
        end
    end
end

