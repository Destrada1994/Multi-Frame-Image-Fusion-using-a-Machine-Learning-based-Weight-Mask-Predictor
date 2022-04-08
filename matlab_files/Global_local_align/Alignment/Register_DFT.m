function [LR_reg, Tvec] = Register_DFT(stack_in,bfn,keep_dim)
resFactor=8;
if bfn ~=1
    first=squeeze(stack_in(:,:,:,1));
    base=squeeze(stack_in(:,:,:,bfn));
    stack_in(:,:,:,1)=base;
    stack_in(:,:,:,bfn)=first;
end

if size(stack_in,3)==3
    for i = 1:size(stack_in,4)
        stack(:,:,i)=rgb2gray(squeeze(stack_in(:,:,:,i)));
    end
else
    stack=squeeze(stack_in);
end

% Get baseframe
baseFrame = squeeze(stack(:,:,1));
height = size(baseFrame,1);
width = size(baseFrame,2);
numImages = size(stack, 3);

baseFrameFFT=fft2(baseFrame);

tforms=affine2d(eye(3));
Tvec{1}=tforms;
for i=2:numImages
    nextFrame=stack(:,:,i);

    [output, ~] = dftregistration(baseFrameFFT,fft2(nextFrame),resFactor);

    tforms(i) = affine2d([ 1 0 0; 0 1 0; output(3) output(4) 1]);
    Tvec{i}=tforms(i);
end


for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 width*resFactor], [1 height*resFactor]);
end

% Find the minimum and maximum output limits.
if keep_dim==1
    xMin=1;
    xMax=width;
    yMin=1;
    yMax=height;
else
    xMin = min([1; xlim(:)]);
    xMax = max([width; xlim(:)]);

    yMin = min([1; ylim(:)]);
    yMax = max([height; ylim(:)]);
end
% Width and height of panorama.
widthA  = round(xMax - xMin+1);
heightA = round(yMax - yMin+1);

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
AlignedView = imref2d([heightA widthA], xLimits, yLimits);

for i=1:size(stack, 3)
    % Warp the current frame using the calculated transformation matrix
    I = imwarp(squeeze(stack_in(:,:,:,i)), tforms(i), 'cubic', 'FillValues', 0,'OutputView', AlignedView);

    % Crop Image to initial size
    LR_reg(:,:,:,i) = I;%(I-min(I(:)))/(max(I(:))-min(I(:)));
end
if bfn ~=1
    first=LR_reg(:,:,:,bfn);
    first_T=Tvec{bfn};
    base=LR_reg(:,:,:,1);
    base_T=Tvec{1};
    LR_reg(:,:,:,1)=first;
    LR_reg(:,:,:,bfn)=base;
    Tvec{1}=first_T;
    Tvec{bfn}=base_T;
end
end