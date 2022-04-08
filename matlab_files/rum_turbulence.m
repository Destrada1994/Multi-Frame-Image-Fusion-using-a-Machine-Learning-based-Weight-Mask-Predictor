clc; clear;

img=imread('C:\Users\denni\FAU_HBOI\MFSR_code\images\target\Scan_00001\image_00001.png');
img_in=double(img)/255;

param.D=0.01; %Aperture diameter in meters
param.d=0.05; %Focal length in meters
param.pixel_sep=1*10^-6; %pixel seperation of sensor
param.lambda=550*10^-9; %Wavelength
param.N=15; %Size of PSF filter
param.L = 5; %Distance from target
param.Cn2 = 1e-10; %refractivity structure index parameter
param.N_psf_gen=64; %Number of PSF to generate


img_out=sim_turbulence(img_in,param);

figure(1)
subplot(1,2,1)
imshow(img_in)
title('Target Image')
subplot(1,2,2)
imshow(img_out)
title(['Degraded Image: C_n^2=',num2str(param.Cn2)])