clc; clear;

filedir='F:\RaspiCam_docs\image_datasets\RaspiCam_Mansion_House\Mansion_1\M10_D30_H7_m50';
DIR=dir(fullfile(filedir,'*.png'));
nfiles=length(DIR);

for i = 1:nfiles
    fid=DIR(i).name;
    img=imread(fullfile(filedir,fid));
    imgs_orig(:,:,:,i)=double(img)/255;
end

scale_in=4;
nfiles=25;
for i = 1:nfiles
    img=imresize(imgs_orig(:,:,:,i),1/scale_in);
    imgs(:,:,:,i)=img;
end

H=size(imgs,1);
W=size(imgs,2);
C=size(imgs,3);
N=size(imgs,4);


%Center Crop 
% crop_sizey=2^(nextpow2(H)-1);
% crop_sizex=2^(nextpow2(W)-1);
% Y_start=H/2-crop_sizey/2;
% X_start=W/2-crop_sizex/2;
% for i=1:N
%     imgs_in(:,:,:,i)=imcrop(imgs(:,:,:,i),[X_start Y_start crop_sizex-1 crop_sizey-1]);
% end
imgs_in=imgs;

H=size(imgs_in,1);
W=size(imgs_in,2);

figure(1)
montage(imgs_in)
title('Captured Images')

baseFrame=imgs_in(:,:,:,1);
negatives=abs(imgs_in-baseFrame);

figure(2)
montage(negatives)
title('Frame Difference from Base Frame Without Alignement')

[imgs_align_LR, T_vec]=RegisterImageSeq(imgs_in);

figure(3)
montage(imgs_align_LR)
title('LR Aligned Images')

LR_fuse_mean=mean(imgs_align_LR,4);

figure(4)
imshow(LR_fuse_mean)
title('Average LR Aligned Image')

masks_LR=dynamicMask(imgs_align_LR);
masks_LR=masks_LR+10^-25;
masks_LR=masks_LR./sum(masks_LR,3);

figure(5)
montage(masks_LR)
title('LR Aligned Image Masks')

H=size(imgs_align_LR,1);
W=size(imgs_align_LR,2);
LR_fused_mask=zeros(H,W,C);
for i=1:N
    for j=1:C
        LR_fused_mask(:,:,j)=LR_fused_mask(:,:,j)+(squeeze(imgs_align_LR(:,:,j,i)).*squeeze(masks_LR(:,:,i)));
    end
end

figure(6)
imshow(LR_fused_mask)
title('LR Fused Image')

resFactor=2;
[imgs_align_HR,Map] = RobustUpSample(imgs_in, T_vec, resFactor);

figure(7)
montage(imgs_align_HR)
title('HR Aligned Images')

HR_fuse_mean=mean(imgs_align_HR,4);

figure(8)
imshow(HR_fuse_mean)
title('Average HR Aligned Image')

masks_HR=dynamicMask(imgs_align_HR).*Map;
masks_HR=masks_HR+10^-25;
masks_HR=masks_HR./sum(masks_HR,3);

figure(9)
montage(masks_HR)
title('HR Aligned Image Masks')

H=size(imgs_align_HR,1);
W=size(imgs_align_HR,2);
HR_fused_mask=zeros(H,W,C);
for i=1:N
    for j=1:C
        HR_fused_mask(:,:,j)=HR_fused_mask(:,:,j)+(squeeze(imgs_align_HR(:,:,j,i)).*squeeze(masks_HR(:,:,i)));
    end
end

figure(10)
imshow(HR_fused_mask)
title('HR Fused Image')

HR_medshift = MedShiftSR(imgs_in, T_vec, resFactor,1);

figure(11)
imshow(HR_medshift)
title('Shift-Median nan-impainting HR Image')

