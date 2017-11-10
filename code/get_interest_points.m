% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width,scale)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.
%This alpha value can be used to tune the feature point detection
alpha=0.06;
%making the edges 0 so that there are no dimension errors
image(1:16,:) = 0;
image(end-16+(1:16),:) = 0;
image(1:16,:) = 0;
image(end-16+(1:16),:) = 0;
image(:,1:16) = 0;
image(:,end-16+(1:16)) = 0;
image(:,1:16) = 0;
image(:,end-16+(1:16)) = 0;
filter_gauss = fspecial('Gaussian', [5 7], 3);
gauss2 = fspecial('Gaussian', [30 30], 1);
%blurring the image(optional)
image_new = imfilter(image, filter_gauss);
%computing the gradients
[der_x,der_y] = imgradientxy(image_new,'sobel');
%double derivatives with the weight function being the guassian filter
der_xx = imfilter(der_x.*der_x, gauss2,'same');
der_yy = imfilter(der_y.*der_y, gauss2,'same');
der_xy = imfilter(der_x.*der_y, gauss2,'same');
%the result of the harris corner detector
r = der_xx.*der_yy - der_xy.*der_xy - alpha.*(der_xx+der_yy).*(der_xx+der_yy);
%threshold to limit the number of interest points detected
%threshold_nd = 0.001; 
%max(max(r))
%min(min(r))
threshold = 0.001*scale;
[y,x] = find(r>threshold);
confidence = r(r>0);
end



