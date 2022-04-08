clc; clear;

%     params  -  object containing all parameters
%     params.t_params  -  object containing turbulence parameters
params.t_params.d = 0.01;     %-  focal length (m) {1.2}
params.t_params.D = 0.005;     %-  aperture diameter size (m) {0.2034}
params.t_params.lambda = 0.525e-6;%-  wavelength (m) {0.525e-6}
params.t_params.L = 5;     %-  propagation length (m) {7000}
params.t_params.Cn2 =3e-10;   %-  index of refraction structure
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


img_in  = im2double(imread('E:\matlab_code\Turbulance_MF\SF_Turbulance\images\Target_usaf\USAF-1951.png'));
% if size(img_in,3)~=1
%     img_in  = rgb2gray(img_in);
% end
img = imresize(img_in,[1024/2,1024/2]);

temp = sim_fun(img,params);

out_stack = uint8(256*temp);

figure(1)
imshow(out_stack)

imwrite(out_stack,'img_mid_turbidity.png')
