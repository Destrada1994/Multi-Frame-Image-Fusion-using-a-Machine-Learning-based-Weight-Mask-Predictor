function [LR_reg, Tvec] = RegisterDFT(stack_in,resFactor)
if size(stack_in,3)==3
    for i = 1:size(stack_in,4)
        stack(:,:,i)=rgb2gray(squeeze(stack_in(:,:,:,i)));
    end
else
    stack=stack_in;
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
    
    tforms(i) = affine2d([ 1 0 0; 0 1 0; output(4) output(3) 1]);
    Tvec{i}=tforms(i);
end


for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 width*resFactor], [1 height*resFactor]);
end

xLimits = [1 width];
yLimits = [1 height];
AlignedView = imref2d([height width], xLimits, yLimits);

for i=1:size(stack, 3)
    % Warp the current frame using the calculated transformation matrix
    %img_in=imresize(squeeze(stack_in(:,:,:,i)),resFactor);
    I = imwarp(stack_in(:,:,:,i), tforms(i), 'cubic', 'FillValues', 1,'OutputView', AlignedView);
    
    % Crop Image to initial size
    LR_reg(:,:,:,i) = I;%(I-min(I(:)))/(max(I(:))-min(I(:)));
end

end