% Matlab Image Feature Registration
%
% Inputs:
% stack - A sequence of low resolution images
%
% Outputs:
% LR_reg - The low resolution stack with registered images
function [LR_reg, Tvec] =RegisterFeatures(stack_in)
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

xLimits = [0 width];
yLimits = [0 height];
AlignedView = imref2d([height width], xLimits, yLimits);

% Initialize features for I(1)
basepoints = detectMinEigenFeatures(baseFrame);
[basefeatures, basepoints] = extractFeatures(baseFrame,basepoints);

tforms=affine2d(eye(3));
Tvec{1}=tforms;
% Iterate through all frames except the base frame
for i=2:numImages
    nextFrame=stack(:,:,i);
    
    % Detect and extract features for I(n).
    points = detectMinEigenFeatures(nextFrame);
    [features, points] = extractFeatures(nextFrame, points);
    
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, basefeatures,'Method','Approximate','MatchThreshold',20,'MaxRatio',1, 'Unique', true);
    
    matchedPoints = points(indexPairs(:,1), :);
    basematchedPoints = basepoints(indexPairs(:,2), :);
    
    % Estimate the transformation between I(n) and I(n-1).
    tforms(i) = estimateGeometricTransform2D(matchedPoints, basematchedPoints,...
        'affine', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    Tvec{i} = tforms(i);
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

