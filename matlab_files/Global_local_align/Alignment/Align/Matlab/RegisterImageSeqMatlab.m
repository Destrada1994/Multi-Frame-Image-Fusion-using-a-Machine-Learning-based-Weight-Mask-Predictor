% Matlab Image Registration
%
% Inputs:
% app - instance of the main app
% stack - A sequence of low resolution images
%
% Outputs:
% LR_reg - The low resolution stack with registered images
% Tvec - The tranlational motion for each LR frame
function [LR_reg, Tvec]=RegisterImageSeqMatlab(stack_in)
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
    
    xLimits = [0 width];
    yLimits = [0 height];
    AlignedView = imref2d([height width], xLimits, yLimits);
    
    
    
    % Init matlab image registration
    [optimizer, metric] = imregconfig('monomodal');
    optimizer.MaximumIterations = 300;

    tforms=affine2d(eye(3));
    Tvec{1}=tforms;
    % Iterate through all frames except the base frame
    for i=2:size(stack, 3)
        % Get transformation matrix from matlab image registration
        tforms(i) = imregtform(squeeze(stack(:,:,i)), baseFrame, 'affine', optimizer, metric);

        % Extract the translation vector and set translation zero
        Tvec{i} = tforms(i);
        %tform.T(3,1:2) = [0 0];
        
    end
    
    for i = 1:numel(tforms)           
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 width], [1 height]);
    end

    % Find the minimum and maximum output limits.
    xMin = min([1; xlim(:)]);
    xMax = max([width; xlim(:)]);
    
    yMin = min([1; ylim(:)]);
    yMax = max([height; ylim(:)]);
    
    % Width and height of panorama.
    widthA  = round(xMax - xMin+1);
    heightA = round(yMax - yMin+1);
    
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
    
end

