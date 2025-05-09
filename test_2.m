%% Advanced Tumor Boundary Tracing and Stiffness Mapping
% This script implements:
% 1. Precise tracing of irregular tumor boundaries using active contours
% 2. Quantitative tissue stiffness mapping with color-coded visualization

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

% Prompt user to select an image (with input validation)
validSelection = false;
while ~validSelection
    selectedIdx = input('Enter the number of the image you want to process: ');
    
    % Validate the input
    if ~isnumeric(selectedIdx) || isempty(selectedIdx)
        disp('Please enter a numeric value.');
    elseif selectedIdx < 1 || selectedIdx > length(files)
        disp(['Please enter a number between 1 and ', num2str(length(files)), '.']);
    else
        validSelection = true;
    end
end

% If no files found, provide a clear error message
if isempty(files)
    error('No image files found in the specified directory. Please check the path and file extension.');
end

% Get the selected image path
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
figure('Name', 'Advanced Tumor Analysis', 'Position', [100, 100, 1200, 800]);
subplot(2, 3, 1);
imshow(img, []);
title('Original Ultrasound Image');

%% Step 2: Apply AGID for noise reduction
% Parameters for anisotropic diffusion
numIterations = 20;
kappa = 15;
lambda = 0.25;
option = 2; % Edge-stopping function

% Apply anisotropic diffusion
filteredImg = anisotropicGrayscaleIntensityDiffusion(img, numIterations, kappa, lambda, option);

% Display filtered image
subplot(2, 3, 2);
imshow(filteredImg, []);
title('AGID Filtered Image');

%% Step 3: Initial tumor segmentation for boundary tracing
% Apply adaptive thresholding
T = adaptthresh(filteredImg, 0.6, 'NeighborhoodSize', 15);
binaryMask = imbinarize(filteredImg, T);

% Clean up binary mask
binaryMask = imfill(binaryMask, 'holes');
binaryMask = bwareaopen(binaryMask, 100);
binaryMask = imopen(binaryMask, strel('disk', 3));
binaryMask = imclose(binaryMask, strel('disk', 5));

% Display initial segmentation
subplot(2, 3, 3);
imshow(binaryMask, []);
title('Initial Tumor Segmentation');

%% Step 4: Active contour for precise boundary tracing
% Use active contours (snakes) to refine the tumor boundary
iterations = 100;
refinedMask = activecontour(filteredImg, binaryMask, iterations, 'edge', 'SmoothFactor', 1.5);

% Find the boundaries of the refined mask
[B, L] = bwboundaries(refinedMask, 'noholes');

% Calculate irregularity metrics for each boundary
irregularityMetrics = zeros(length(B), 1);
for k = 1:length(B)
    boundary = B{k};
    % Calculate perimeter
    perimeter = size(boundary, 1);
    % Calculate area
    area = regionprops(L==k, 'Area').Area;
    % Circularity/irregularity metric (1 for perfect circle, higher for irregular)
    irregularityMetrics(k) = perimeter^2 / (4 * pi * area);
end

% Display refined segmentation with boundary
subplot(2, 3, 4);
imshow(filteredImg, []);
hold on;
for k = 1:length(B)
    boundary = B{k};
    % Colormap based on irregularity (red = more irregular)
    color = [1, 1 - min(1, (irregularityMetrics(k)-1)/5), 0];
    plot(boundary(:,2), boundary(:,1), 'LineWidth', 2, 'Color', color);
end
title('Traced Tumor Boundaries');
colormap(gca, jet);

%% Step 5: Tissue stiffness estimation using multiple texture features
% Create empty stiffness map
stiffnessMap = zeros(size(filteredImg));
windowSize = 15;
halfSize = floor(windowSize/2);

% Set of texture features for stiffness estimation
for i = halfSize+1:size(filteredImg,1)-halfSize
    for j = halfSize+1:size(filteredImg,2)-halfSize
        % Extract window
        window = filteredImg(i-halfSize:i+halfSize, j-halfSize:j+halfSize);
        
        % Calculate multiple texture features
        % 1. Standard deviation (heterogeneity)
        stdDev = std2(window);
        
        % 2. Local entropy (complexity)
        entropy = entropyfilt(window, true(3));
        entropy = mean2(entropy);
        
        % 3. Local Range (contrast)
        range = max(window(:)) - min(window(:));
        
        % Combine features for stiffness estimation (weighted sum)
        stiffnessMap(i,j) = 0.4*stdDev + 0.4*entropy + 0.2*range;
    end
end

% Normalize stiffness map
stiffnessMap = stiffnessMap / max(stiffnessMap(:));

% Apply median filtering to smooth the stiffness map
stiffnessMap = medfilt2(stiffnessMap, [5 5]);

% Show stiffness map
subplot(2, 3, 5);
imshow(stiffnessMap, []);
colormap(gca, parula);
colorbar;
title('Tissue Stiffness Map');

%% Step 6: Create comprehensive visualization with quantitative metrics
% Create a combined visualization
subplot(2, 3, 6);
imshow(filteredImg, []);
hold on;

% Overlay stiffness as transparent color map
h = imshow(stiffnessMap, []);
colormap(gca, parula);
set(h, 'AlphaData', stiffnessMap * 0.7);

% Add boundary traces
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'LineWidth', 2, 'Color', 'r');
    
    % Add irregularity metric text near each boundary
    centroid = regionprops(L==k, 'Centroid').Centroid;
    text(centroid(1), centroid(2), sprintf('IR: %.2f', irregularityMetrics(k)), ...
        'Color', 'white', 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.5]);
end

% Add colorbar for stiffness interpretation
colorbar;
title('Combined View: Boundaries & Stiffness');

%% Step 7: Generate a detailed report figure
figure('Name', 'Tumor Analysis Report', 'Position', [100, 100, 800, 600]);

% Plot original image with tumor boundary
subplot(1, 2, 1);
imshow(filteredImg, []);
hold on;
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'LineWidth', 2, 'Color', 'r');
end
title('Tumor Boundary Tracing');

% Calculate and display quantitative metrics
subplot(1, 2, 2);

% Create stiffness histogram within tumor region
tumorStiffness = stiffnessMap(refinedMask);
histogram(tumorStiffness, 20, 'Normalization', 'probability');
title('Stiffness Distribution Within Tumor');
xlabel('Relative Stiffness');
ylabel('Frequency');
grid on;

% Add text annotations with metrics
annotation('textbox', [0.5, 0.1, 0.45, 0.3], 'String', {...
    sprintf('Mean Tumor Stiffness: %.3f', mean(tumorStiffness)), ...
    sprintf('Stiffness Variance: %.3f', var(tumorStiffness)), ...
    sprintf('Average Boundary Irregularity: %.3f', mean(irregularityMetrics)), ...
    sprintf('Tumor Area: %d pixels', sum(refinedMask(:))), ...
    sprintf('Tumor Perimeter: %d pixels', sum(cell2mat(cellfun(@length, B, 'UniformOutput', false))))...
    }, 'EdgeColor', 'none', 'BackgroundColor', [0.9 0.9 0.9]);

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