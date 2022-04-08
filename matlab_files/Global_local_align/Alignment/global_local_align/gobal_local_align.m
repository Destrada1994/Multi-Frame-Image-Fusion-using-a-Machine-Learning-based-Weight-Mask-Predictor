clc; clear;

filedir='C:\Users\denni\Documents\MATLAB\Images\Tietze';
DIR=dir(fullfile(filedir,'*.png'));
nfiles=length(DIR);

scale_in=4;
nfiles=4;
for i = 1:nfiles
    fid=DIR(i).name;
    img=imread(fullfile(filedir,fid));
    img=imresize(img,1/scale_in);
    imgs(:,:,:,i)=double(img)/255;
end

%% Global Alignment

[imgs_align, T_vec]=RegisterImageSeq(imgs);

H=size(imgs_align,1);
W=size(imgs_align,2);
C=size(imgs_align,3);
N=size(imgs_align,4);

crop_sizey=512/2;
crop_sizex=1024/2;
Y_start=H/2-crop_sizey/2;
X_start=W/2-crop_sizex/2;
for i=1:N
    imgs_crop(:,:,:,i)=imcrop(imgs_align(:,:,:,i),[X_start Y_start crop_sizex-1 crop_sizey-1]);
end
H=size(imgs_crop,1);
W=size(imgs_crop,2);

SR_scale=1;

box_size=64;
nh=floor(H/box_size);
nw=floor(W/box_size);

Align_Matrix=local_align(imgs_crop,box_size,nw,nh);

SR1=nan(H*SR_scale,W*SR_scale,C,N);
zero_shift_y=1:SR_scale:H*SR_scale;
zero_shift_x=1:SR_scale:W*SR_scale;

for i=1:H
    for j=1:W
        for k=1:N
            pixels=imgs_crop(i,j,:,k);
            shift=Align_Matrix(i,j,:,k);
            shiftSR=round(shift*SR_scale);
            index_y=zero_shift_y(i);
            index_x=zero_shift_x(j);
            if index_y+shiftSR(1) >0 && index_x+shiftSR(2)>0
                if index_y+shiftSR(1) <H*SR_scale && index_x+shiftSR(2)<W*SR_scale
                    
                    SR1(index_y+shiftSR(1),index_x+shiftSR(2),:,k)=pixels;
                end
            end
        end
    end
end

for i=1:N
    for j=1:C
        SR(:,:,j,i)=inpaint_nans(SR1(:,:,j,i), 0);
    end
end

thresh=0.1;
masks = dynamicMask(SR);
masks(masks<=thresh)=0;
masks=masks+10^-25;
masks=masks./sum(masks,3);

pyr = gaussian_pyramid(zeros(H*SR_scale,W*SR_scale,C));
nlev = length(pyr);

% multiresolution blending
for i = 1:N
    % construct pyramid from each input image
    pyrW = gaussian_pyramid(masks(:,:,i));
    
    pyrI = laplacian_pyramid(SR(:,:,:,i));
    
    % blend
    for b = 1:nlev
        w = repmat(pyrW{b},[1 1 3]);
        
        pyr{b} = pyr{b} + w .* pyrI{b};
    end
end

% reconstruct
HR = reconstruct_laplacian_pyramid(pyr);

figure(1)
montage(imgs_crop)

figure(2)
montage(SR)

figure(3)
montage(masks)

figure(4)
imshow(imgs_crop(:,:,:,1))

figure(5)
imshow(HR)