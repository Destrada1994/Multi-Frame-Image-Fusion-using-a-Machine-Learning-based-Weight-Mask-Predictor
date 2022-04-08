clc; clear;

filedir='E:\Dissertation_MFSR_ML\Code\Multi-Image Super Resolution\Jan2022_MFSR\Images\input_DS\Scan_00001';
DIR=dir(fullfile(filedir,'*.png'));
nfiles=length(DIR);

nfiles=4;
for i = 1:nfiles
    fid=DIR(i).name;
    imgs(:,:,:,i)=double(imread(fullfile(filedir,fid)))/255;
end

AlignMode = 5;

if AlignMode ==0
    imgs_align=imgs;
elseif AlignMode == 1
    [imgs_align, T_vec]=RegisterImageSeqMatlab(imgs);
elseif AlignMode == 2
    [imgs_align, T_vec]=RegisterFeatures(imgs);
elseif AlignMode == 3
    [imgs_align, T_vec]=RegisterDFT(imgs,1);
elseif AlignMode == 4
    [imgs_align, T_vec]=RegisterImageSeq(imgs);
elseif AlignMode == 5
    [imgs_align, T_vec]=RegisterImageSeqAffine(imgs);
end

figure(1)
montage(imgs_align)
title('Multi-Exposure Images')


