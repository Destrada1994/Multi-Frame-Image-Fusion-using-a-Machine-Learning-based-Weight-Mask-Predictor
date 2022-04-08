clc; clear;

filedirT{1}='F:\RaspiCam_docs\image_datasets\RCA_mansion\Test_sets\M11_D2_H15_m33\scan_2_5pixvar_1ISO_4497shutterspeed';
filedirT{2}='F:\RaspiCam_docs\image_datasets\RCA_mansion\Test_sets\M11_D2_H18_m18\scan_1_5pixvar_600ISO_2389shutterspeed';
filedirT{3}='F:\RaspiCam_docs\image_datasets\RCA_mansion\Test_sets\M11_D3_H7_m22\scan_1_5pixvar_600ISO_55227shutterspeed';

for FN=1:3
    filedir=filedirT{FN};
    DIR=dir(fullfile(filedir,'*.png'));
    nfiles=9;%length(DIR);
    
    savedir=fullfile(filedir,'Fusion_results');
    mkdir(savedir)
    
    for i = 1:nfiles
        fid=DIR(i).name;
        img=imread(fullfile(filedir,fid));
        imgs_orig(:,:,:,i)=double(img)/255;
    end
    
    scale_in=1;
    for i = 1:nfiles
        img=imresize(imgs_orig(:,:,:,i),1/scale_in);
        imgs(:,:,:,i)=img;
    end
    
    figure(1)
    montage(imgs)
    title('Captured Images')
    
    H=size(imgs,1);
    W=size(imgs,2);
    C=size(imgs,3);
    N=size(imgs,4);
    
    
    %Center Crop 
    crop_sizey=1028;
    crop_sizex=1028;
    Y_start=H/2-crop_sizey;
    X_start=W-crop_sizex;
    for i=1:N
        imgs_in(:,:,:,i)=imcrop(imgs(:,:,:,i),[X_start Y_start crop_sizex-1 crop_sizey-1]);
    end
    %imgs_in=imgs;
    
    H=size(imgs_in,1);
    W=size(imgs_in,2);
    
    figure(2)
    montage(imgs_in)
    title('Cropped Images')
    
    baseFrame=imgs_in(:,:,:,1);
    negatives=abs(imgs_in-baseFrame);
    
    figure(3)
    montage(negatives)
    title('Frame Difference from Base Frame Without Alignement')
    
    [imgs_align_LR, T_vec]=RegisterImageSeq(imgs_in);
%     imgs_align_LR=imgs_in;
%     for i=1:nfiles
%         T_vec{i}=affine2d(eye(3));
%     end
    
    baseFrame=imgs_align_LR(:,:,:,1);
    negatives=abs(imgs_align_LR-baseFrame);
    
    figure(4)
    montage(negatives)
    title('Frame Difference from Base Frame With Alignement')
    
    figure(5)
    imshow(imgs_in(:,:,:,1))
    title('Input Base Frame')
    imwrite(imgs_in(:,:,:,1),fullfile(savedir,'Base_Frame.png'))

    LR_fuse_mean=mean(imgs_align_LR,4);
    
    figure(6)
    imshow(LR_fuse_mean)
    title('Average LR Aligned Image')
    imwrite(LR_fuse_mean,fullfile(savedir,'LR_fused_mean.png'))
    
    masks_LR=dynamicMask(imgs_align_LR);
    masks_LR=masks_LR+10^-25;
    masks_LR=masks_LR./sum(masks_LR,3);
    
    figure(7)
    montage(masks_LR)
    title('LR Aligned Image Masks')
    
    
    LR_fused_mask=pyramidBlend_mask(imgs_align_LR,masks_LR);
    
    figure(8)
    imshow(LR_fused_mask)
    title('LR Fused Image')
    imwrite(LR_fused_mask,fullfile(savedir,'LR_fused_mask.png'))

    resFactor=2;
    [imgs_align_HR,Map] = RobustUpSample(imgs_in, T_vec, resFactor);
    
    figure(9)
    imshow(imgs_align_HR(:,:,:,1))
    title('Base Frame Bicubic Upsample')
    imwrite(imgs_align_HR(:,:,:,1),fullfile(savedir,'Base_Frame_Bicubic.png'))
    
    HR_fuse_mean=mean(imgs_align_HR,4);
    
    figure(10)
    imshow(HR_fuse_mean)
    title('Average HR Aligned Image')
    imwrite(HR_fuse_mean,fullfile(savedir,'HR_fused_mean.png'))

    masks_HR=dynamicMask(imgs_align_HR);
    masks_HR=masks_HR+10^-25;
    masks_HR=masks_HR./sum(masks_HR,3);
    
    figure(11)
    montage(masks_HR)
    title('HR Aligned Image Masks')
    
    HR_fused_mask=pyramidBlend_mask(imgs_align_HR,masks_HR);
    
    figure(12)
    imshow(HR_fused_mask)
    title('HR Fused Image')
    imwrite(HR_fused_mask,fullfile(savedir,'HR_fused_mask.png'))
    
%     HR_medshift = MedShiftSR(imgs_in, T_vec, resFactor,1);
%     
%     figure(13)
%     imshow(HR_medshift)
%     title('Shift-Median nan-impainting HR Image')
%     imwrite(HR_medshift,fullfile(savedir,'HR_medshift.png'))
end

