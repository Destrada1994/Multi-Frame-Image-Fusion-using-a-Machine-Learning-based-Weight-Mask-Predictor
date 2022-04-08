clc; clear;

filedir='C:\Users\denni\Documents\MATLAB\Images\Tietze';
DIR=dir(fullfile(filedir,'*.png'));
nfiles=length(DIR);

scale_in=2;
nfiles=9;
for i = 1:nfiles
    fid=DIR(i).name;
    img=imread(fullfile(filedir,fid));
    img=imresize(img,1/scale_in);
    imgs(:,:,:,i)=double(img)/255;
end

%% Global Alignment

[imgs_align, T_vec]=RegisterImageSeq(imgs);

figure(1)
montage(imgs)
title('Captured Images')

figure(2)
montage(imgs_align)
title('Globally Aligned Images')

H=size(imgs_align,1);
W=size(imgs_align,2);
C=size(imgs_align,3);
N=size(imgs_align,4);

baseFrame=imgs(:,:,:,1); %Base Frame is the Middle Frame
negatives=abs(imgs-baseFrame);

figure(3)
montage(negatives)
title('Frame Difference from Base Frame Without Global Alignement')

baseFrame_align=imgs_align(:,:,:,1); %Base Frame is the Middle Frame
negatives_align=abs(imgs_align-baseFrame_align);

figure(4)
montage(negatives_align)
title('Frame Difference from Base frame With Global Alignment')

%Center Crop 
crop_sizey=2^(nextpow2(H)-1);
crop_sizex=2^(nextpow2(W)-1);
Y_start=H/2-crop_sizey/2;
X_start=W/2-crop_sizex/2;
for i=1:N
    imgs_crop(:,:,:,i)=imcrop(imgs_align(:,:,:,i),[X_start Y_start crop_sizex-1 crop_sizey-1]);
end

figure(5)
montage(imgs_crop)
title('Center Crop if Image Stack')

SR_scale=2;
if SR_scale~=1
    for i=1:N
        imgs_in(:,:,:,i)=imresize(imgs_crop(:,:,:,i),SR_scale);
    end
else
    imgs_in=imgs_crop;
end

H=size(imgs_in,1);
W=size(imgs_in,2);

box_size=256;
nh=floor(H/box_size);
nw=floor(W/box_size);

Align_Matrix=local_align2(imgs_in,box_size,nw,nh);

SR1=imgs_in;
A=zeros(H,W,N);
zero_shift_y=1:H;
zero_shift_x=1:W;

for i=1:H
    for j=1:W
        for k=1:N
            pixels=imgs_in(i,j,:,k);
            shift=Align_Matrix(i,j,:,k);
            shiftSR=round(shift);
            if i+shiftSR(1) >0 && j+shiftSR(2)>0
                if i+shiftSR(1) <H && j+shiftSR(2)<W
                    SR1(i+shiftSR(1),j+shiftSR(2),:,k)=pixels;
                    A(i+shiftSR(1),j+shiftSR(2),k)=1;
                end
            end
        end
    end
end

figure(6)
montage(SR1)
title('Localized Alignment of Center Crop')

baseFrame_align_local=SR1(:,:,:,1); %Base Frame is the Middle Frame
negatives_align_local=abs(SR1-baseFrame_align_local);

figure(7)
montage(negatives_align_local)
title('Frame Difference from Base frame With Global & Local Alignment')

masks=dynamicMask(SR1);
masks_round=round(masks);

figure(8)
montage(masks)
title('Image Masks')

SR_fused=zeros(H,W,C);
for j=1:H
    for k=1:W
        for c=1:C
            SR_fused(j,k,c)=median(SR1(j,k,c,masks_round(j,k,:)==1));
        end
    end
end

figure(9)
imshow(imgs_in(:,:,:,1))

figure(10)
imshow(SR_fused)

masks_norm=masks+10^-25;
masks_norm=masks_norm./sum(masks_norm,3);

% create empty pyramid
pyr = gaussian_pyramid(zeros(H,W));
nlev = length(pyr);

% multiresolution blending
for i = 1:N
    % construct pyramid from each input image
    pyrW = gaussian_pyramid(masks_norm(:,:,i));
    
    pyrI = laplacian_pyramid(SR1(:,:,:,i));
    
    % blend
    for b = 1:nlev
        w = repmat(pyrW{b},[1 1 3]);
        
        pyr{b} = pyr{b} + w .* pyrI{b};
    end
end

% reconstruct
HR_pyramid = reconstruct_laplacian_pyramid(pyr);

figure(11)
imshow(HR_pyramid)

nlev=3;
wavF=wavelet_decomp_RGB(zeros(H,W,C),nlev);

for i = 1:N
    wavI = wavelet_decomp_RGB(SR1(:,:,:,i).*masks_norm(:,:,i),nlev);
    for j = 1:2^(nlev*2)
        wavF{j}=wavF{j} + wavI{j};
    end
end

HR_wavelet = wavelet_recons_RGB(wavF,nlev);

figure(12)
imshow(HR_wavelet)
