clc; clear;

filedir='C:\Users\denni\FAU_HBOI\Dissertation\Code\images\train\A\scan_00002';
DIR=dir(fullfile(filedir,'*.png'));
nfiles=length(DIR);

nframes=9;
DS_scale=1;
UP_scale=1;
bfn=5;

c=0;
for i=1:nframes
    c=c+1;
    fid=DIR(i).name;
    img=double(imread(fullfile(filedir,fid)))/255;
    img=imresize(img,1/DS_scale);
    stack(:,:,:,c)=img;
end

figure(1)
montage(stack)
title('Input Images')

[align1, global_Tvec]=Register_LKO(stack,bfn,0);
global_align=UpSample_Shift(align1,global_Tvec,UP_scale,2);
%[global_align, global_Tvec]=RegisterImageSeqAffine(stack,5,0);

figure(2)
montage(global_align)
title('Globally Aligned Images')

[local_align,local_Tvec]=local_patch_align(global_align,bfn,64);

figure(3)
montage(local_align)
title('Locally Aligned')

base_frame=global_align(:,:,:,bfn);
diff=abs(global_align-base_frame);

figure(4)
montage(diff)
title('Global Aligned Difference to Base Frame')

base_frame=global_align(:,:,:,bfn);
diff=abs(local_align-base_frame);

figure(5)
montage(diff)
title('Local Aligned Difference to Base Frame')
