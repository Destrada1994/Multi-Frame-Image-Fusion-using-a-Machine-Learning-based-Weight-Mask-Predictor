% Image-Registration of a given sequence of images
%
% This function uses a simplified Lucas-Kanade method to
% register images solely based on their translatory
% deviation. It uses a hierarchical gradient-based
% optimization method, using 6 levels of Low-Pass filtering
% for calculation of the optical flow parameters between
% two images
%
% Inputs:
% app - instance of the main app
% stack - A sequence of low resolution images
%
% Outputs:
% LR_reg - The low resolution stack with registered images
% Tvec - The tranlational motion for each LR frame
% iter - Steps needed for registration
% err - Sum of error during registration
function [LR_reg, Tvec]=RegisterImageSeq(stack_in,bfn,keep_dim)
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
        stack=stack_in;
    end
    % Init variables
    iter = 0; err = 0;
    
    % Get baseframe
    baseFrame = squeeze(stack(:,:,1));
    height = size(baseFrame,1);
    width = size(baseFrame,2);
    
    % Create the region of continuous flow for lowest hierarchy
    % level of the gaussian pyramid
    roi=[2 2 size(stack,1)-1 size(stack,2)-1];
    
    tforms=affine2d(eye(3));
    Tvec{1}=tforms;
    for i=2:size(stack,3)
        % Register current image to previous frame
        [D, k, e] = PyramidalLKOpticalFlow(baseFrame, squeeze(stack(:,:,i)), roi);

        % Project the 2D-motion vector onto the general affine
        % transformation matrix
        if isnan(D(1))
            D(1)=0;
        end
        if isnan(D(2))
            D(2)=0;
        end
        % Perform Image registration
        tforms(i) = affine2d([ 1 0 0; 0 1 0; D(1) D(2) 1]);
        Tvec{i} = tforms(i);
        % Sum up the iterations and errors
        iter = iter + k;
        err = err + e;
    end

    for i = 1:numel(tforms)           
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 width], [1 height]);
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
    widthA  = round(xMax - xMin)+1;
    heightA = round(yMax - yMin)+1;

    % Create a 2-D spatial reference object defining the size of the panorama.
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    AlignedView = imref2d([heightA widthA], xLimits, yLimits);
    
    for i=1:size(stack, 3)
        % Warp the current frame using the calculated transformation matrix
        I = imwarp(squeeze(stack_in(:,:,:,i)), tforms(i), 'cubic', 'FillValues', 1,'OutputView', AlignedView);

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