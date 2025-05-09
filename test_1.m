%% Breast Tumor Image Processing - Boundary and Stiffness Visualization
% This script implements a simplified version of:
% 1. Anisotropic Grayscale Intensity Diffusion (AGID) for noise removal
% 2. Bird-Dipper Throated Feature Extraction (BDTFE) for highlighting
%    irregular tumor boundaries and tissue stiffness

clear all;
close all;
clc;

%% Step 1: Load an image from test folder
% Define the folder path containing test images
testFolderPath = 'D:\Kannadasan\MRL Paper Works\Veeramani correction work\matlab code\test_images'; % MODIFY THIS PATH
% Get list of all images in the folder
filePattern = fullfile(testFolderPath, '*.png'); % Change extension if needed
files = dir(filePattern);

% Display available images and let user select one
disp('Available test images:');
for i = 1:length(files)
    disp([num2str(i), '. ', files(i).name]);
end

% Prompt user to select an image
selectedIdx = input('Enter the number of the image you want to process: ');
selectedImagePath = fullfile(testFolderPath, files(selectedIdx).name);

% Read the selected image
img = imread(selectedImagePath);

% Convert to grayscale if it's RGB
if size(img, 3) > 1
    img = rgb2gray(img);
end

% Convert to double for processing
img = double(img);
img = img / max(img(:)); % Normalize to [0, 1]

% Display original image
figure('Name', 'Breast Ultrasound Image Analysis');
subplot(2, 2, 1);
imshow(img, []);
title('Original Ultrasound Image');

%% Step 2: Implement simplified AGID for noise reduction
% Parameters for anisotropic diffusion
numIterations = 15;
kappa = 15;
lambda = 0.25;
option = 2; % Edge-stopping function (1 or 2)

% Anisotropic diffusion function (modified Perona-Malik model for AGID)
filteredImg = anisotropicGrayscaleIntensityDiffusion(img, numIterations, kappa, lambda, option);

% Display filtered image
subplot(2, 2, 2);
imshow(filteredImg, []);
title('AGID Filtered Image');

%% Step 3: Extract and visualize tumor boundaries using simplified BDTFE
% Edge detection for boundary identification
[Gmag, Gdir] = imgradient(filteredImg, 'sobel');

% Apply adaptive thresholding to identify strong edges
threshBoundary = adaptthresh(Gmag, 0.4, 'ForegroundPolarity', 'bright');
boundaryMask = Gmag > (max(Gmag(:)) * threshBoundary);

% Clean up the boundary mask
boundaryMask = bwareaopen(boundaryMask, 20);
boundaryMask = imclose(boundaryMask, strel('disk', 2));

% Create boundary highlight image
boundaryImg = filteredImg;
boundaryImg(boundaryMask) = 1; % Highlight boundaries

% Display boundary image
subplot(2, 2, 3);
imshow(boundaryImg, []);
title('Tumor Boundary Visualization');
colormap(gca, jet); % Apply color to highlight boundaries

%% Step 4: Extract and visualize tissue stiffness using texture analysis
% Compute GLCM for texture analysis (simplified stiffness estimation)
glcm = graycomatrix(im2uint8(filteredImg), 'Offset', [0 1; -1 1; -1 0; -1 -1], 'NumLevels', 32, 'Symmetric', true);
    
% Extract texture features
stats = graycoprops(glcm, {'contrast', 'homogeneity', 'energy', 'correlation'});

% Create a stiffness map using contrast (higher contrast indicates different stiffness)
stiffnessMap = zeros(size(filteredImg));
windowSize = 11;
halfSize = floor(windowSize/2);

% Compute local contrast in sliding window
for i = halfSize+1:size(filteredImg,1)-halfSize
    for j = halfSize+1:size(filteredImg,2)-halfSize
        window = filteredImg(i-halfSize:i+halfSize, j-halfSize:j+halfSize);
        stiffnessMap(i,j) = std2(window) * 5; % Amplify for visualization
    end
end

% Normalize stiffness map
stiffnessMap = stiffnessMap / max(stiffnessMap(:));

% Apply color mapping to stiffness
subplot(2, 2, 4);
imshow(stiffnessMap, []);
colormap(gca, hot); % Use hot colormap for stiffness visualization
colorbar;
title('Tissue Stiffness Visualization');

%% Display combined visualization in a new figure
figure('Name', 'Combined Tumor Analysis');
% Create color overlay of boundaries on original image
rgb = repmat(filteredImg, [1 1 3]); % Convert to RGB
boundaryOverlay = boundaryMask .* 0.8; % Scale intensity for visualization

% Add boundaries in blue
rgb(:,:,3) = rgb(:,:,3) + boundaryOverlay;
rgb(rgb > 1) = 1; % Ensure values stay in valid range

% Add stiffness in red (where stiffness is high)
stiffnessThresh = stiffnessMap > 0.5;
rgb(:,:,1) = rgb(:,:,1) + (stiffnessThresh .* stiffnessMap * 0.7);
rgb(rgb > 1) = 1;

% Display final result
imshow(rgb);
title('Combined View: Tumor Boundaries (Blue) and Tissue Stiffness (Red)');

% Add colorbar for stiffness interpretation
c = colorbar;
c.Label.String = 'Relative Stiffness';

%% Function implementations
function diffused = anisotropicGrayscaleIntensityDiffusion(img, numIter, kappa, lambda, option)
    % Implementation of Anisotropic Grayscale Intensity Diffusion (AGID)
    % for speckle and artifact reduction in ultrasound images
    
    % Initialize
    diffused = img;
    [rows, cols] = size(img);
    
    % Directional derivatives
    dx = zeros(rows, cols, numIter+1);
    dy = zeros(rows, cols, numIter+1);
    
    % Initialize diffusion coefficient matrices
    c1 = zeros(rows, cols);
    c2 = zeros(rows, cols);
    c3 = zeros(rows, cols);
    c4 = zeros(rows, cols);
    
    % Store initial image
    dx(:,:,1) = diffused;
    dy(:,:,1) = diffused;
    
    % Main diffusion loop
    for i = 1:numIter
        % Calculate gradients
        [dN, dS, dE, dW] = getNeighborDifferences(diffused);
        
        % Calculate diffusion coefficients
        if option == 1
            % Perona-Malik diffusion coefficient 1
            c1 = 1 ./ (1 + (dN/kappa).^2);
            c2 = 1 ./ (1 + (dS/kappa).^2);
            c3 = 1 ./ (1 + (dE/kappa).^2);
            c4 = 1 ./ (1 + (dW/kappa).^2);
        elseif option == 2
            % Perona-Malik diffusion coefficient 2
            % Better preserves edges than option 1
            c1 = exp(-(dN/kappa).^2);
            c2 = exp(-(dS/kappa).^2);
            c3 = exp(-(dE/kappa).^2);
            c4 = exp(-(dW/kappa).^2);
        end
        
        % Modified for AGID - include intensity weighting
        intensity_weight = 1 + 5*(diffused - min(diffused(:)))/(max(diffused(:)) - min(diffused(:)));
        
        % Apply diffusion update
        diffused = diffused + lambda * (c1.*dN + c2.*dS + c3.*dE + c4.*dW) .* intensity_weight;
        
        % Store result
        dx(:,:,i+1) = diffused;
        dy(:,:,i+1) = diffused;
    end
end

function [dN, dS, dE, dW] = getNeighborDifferences(img)
    % Calculate differences between neighboring pixels
    [rows, cols] = size(img);
    
    % North, South, East, West differences
    dN = zeros(rows, cols);
    dS = zeros(rows, cols);
    dE = zeros(rows, cols);
    dW = zeros(rows, cols);
    
    % North pixel difference (j, i-1)
    dN(2:rows, :) = img(1:rows-1, :) - img(2:rows, :);
    
    % South pixel difference (j, i+1)
    dS(1:rows-1, :) = img(2:rows, :) - img(1:rows-1, :);
    
    % East pixel difference (j+1, i)
    dE(:, 1:cols-1) = img(:, 2:cols) - img(:, 1:cols-1);
    
    % West pixel difference (j-1, i)
    dW(:, 2:cols) = img(:, 1:cols-1) - img(:, 2:cols);
end