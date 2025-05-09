%% Breast Tumor Elastography Visualization
% This script creates an elastography-style visualization that shows:
% 1. Original grayscale ultrasound image 
% 2. Elastography overlay with kPa scale (tissue stiffness)
% 3. Side-by-side display with color scale bar

clear all;
close all;
clc;

%% Step 1: Load a test image from the specified folder
% Define the folder path containing test images
testFolderPath = 'D:\Kannadasan\MRL Paper Works\Veeramani correction work\matlab code\test_images';
disp(['Looking for images in: ', testFolderPath]);

% Get list of all images in the folder (support multiple image formats)
pngFiles = dir(fullfile(testFolderPath, '*.png'));
jpgFiles = dir(fullfile(testFolderPath, '*.jpg'));
bmpFiles = dir(fullfile(testFolderPath, '*.bmp'));
tifFiles = dir(fullfile(testFolderPath, '*.tif'));
files = [pngFiles; jpgFiles; bmpFiles; tifFiles];

% Check if any files were found
if isempty(files)
    error('No image files found in the specified directory. Please check the path and file extensions.');
end

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

% Get the selected image path
selectedImagePath = fullfile(testFolderPath, files(selectedIdx).name);
disp(['Processing image: ', files(selectedIdx).name]);

%% Step 2: Read and preprocess the selected image
% Read the selected image
img = imread(selectedImagePath);

% Convert to grayscale if it's RGB
if size(img, 3) > 1
    img = rgb2gray(img);
    disp('Converted RGB image to grayscale');
end

% Convert to double for processing and normalize
img = double(img);
img = img / max(img(:)); % Normalize to [0, 1]

%% Step 3: Apply noise reduction
disp('Applying noise reduction...');
filteredImg = anisotropicDiffusion(img, 20, 0.15, 0.9, 2);

%% Step 4: Detect region of interest (ROI) for elastography
disp('Detecting region of interest...');

% Invert image to make dark regions (potential tumors) bright
invertedImg = 1 - filteredImg;

% Apply adaptive thresholding to identify dark regions
T = adaptthresh(invertedImg, 0.6, 'NeighborhoodSize', 25);
regionMask = imbinarize(invertedImg, T);

% Clean up binary mask
regionMask = imfill(regionMask, 'holes');
regionMask = bwareaopen(regionMask, 200); % Remove small regions
regionMask = imopen(regionMask, strel('disk', 2));
regionMask = imclose(regionMask, strel('disk', 5));

% If no significant region found, create a central ROI
if sum(regionMask(:)) < 1000
    disp('No clear tumor region detected. Creating manual ROI...');
    [rows, cols] = size(img);
    [X, Y] = meshgrid(1:cols, 1:rows);
    centerX = cols/2;
    centerY = rows/2;
    radius = min(rows, cols)/4;
    regionMask = ((X-centerX).^2 + (Y-centerY).^2) <= radius^2;
end

%% Step 5: Calculate elastography metrics (simulating tissue stiffness)
disp('Calculating tissue stiffness metrics...');

% Create empty stiffness map
stiffnessMap = zeros(size(filteredImg));
windowSize = 15;
halfSize = floor(windowSize/2);

% Simulate stiffness calculation based on multiple texture features
for i = halfSize+1:size(filteredImg,1)-halfSize
    for j = halfSize+1:size(filteredImg,2)-halfSize
        % Skip computation for non-ROI areas
        if ~any(any(regionMask(i-halfSize:i+halfSize, j-halfSize:j+halfSize)))
            continue;
        end
        
        % Extract window
        window = filteredImg(i-halfSize:i+halfSize, j-halfSize:j+halfSize);
        
        % Calculate features that correlate with stiffness
        % 1. Local standard deviation - heterogeneity correlates with stiffness
        stdDev = std2(window);
        
        % 2. Local entropy - complexity correlates with stiffness
        entropyVal = entropyfilt(window, true(3));
        entropyVal = mean2(entropyVal);
        
        % 3. Gradients - boundaries between tissues of different stiffness
        [Gx, Gy] = gradient(window);
        gradientMag = mean2(sqrt(Gx.^2 + Gy.^2));
        
        % 4. Local intensity - darker regions often represent harder tissue
        darknessFactor = 1 - mean2(window);
        
        % Combine features with weights to simulate stiffness
        stiffnessMap(i,j) = 150*darknessFactor + 80*stdDev + 50*entropyVal + 20*gradientMag;
    end
end

% Add some spatial coherence to the stiffness map
stiffnessMap = imgaussfilt(stiffnessMap, 2);

% Scale to desired kPa range (0-210 kPa, matching the example image)
stiffnessMap = stiffnessMap / max(stiffnessMap(:)) * 210;

% Set non-ROI areas to NaN to make them transparent in the overlay
stiffnessMapMasked = stiffnessMap;
stiffnessMapMasked(~regionMask) = NaN;

%% Step 6: Create elastography visualization
disp('Creating elastography visualization...');

% Create figure with the layout similar to the example image
figure('Name', 'Breast Ultrasound Elastography', 'Position', [100, 100, 800, 600], 'Color', 'k');

% Display the original grayscale image in the full frame
ax1 = axes('Position', [0, 0, 0.8, 1]);
imshow(filteredImg, []);
colormap(ax1, 'gray');
axis off;

% Create ROI overlay region in top half
ax2 = axes('Position', [0.15, 0.5, 0.5, 0.5]);
imshow(stiffnessMapMasked, [0, 210]);
colormap(ax2, jet);
set(ax2, 'Box', 'on', 'XColor', 'w', 'YColor', 'w');
% Make top left and top right tickmarks visible
set(ax2, 'XTick', [1, size(stiffnessMapMasked, 2)], 'YTick', [1], 'XTickLabel', [], 'YTickLabel', []);

% Draw a white border around the elastography overlay
hold(ax2, 'on');
plot(ax2, [1, size(stiffnessMapMasked, 2), size(stiffnessMapMasked, 2), 1, 1], ...
     [1, 1, size(stiffnessMapMasked, 1), size(stiffnessMapMasked, 1), 1], 'w-', 'LineWidth', 1.5);
hold(ax2, 'off');

% Create a custom colorbar on the right side
ax3 = axes('Position', [0.85, 0.15, 0.1, 0.7]);
colorbarImg = repmat(linspace(0, 1, 256)', 1, 30);
image(ax3, colorbarImg);
colormap(ax3, jet);
axis(ax3, 'off');

% Add tick marks and labels to the colorbar
ax4 = axes('Position', [0.85, 0.15, 0.1, 0.7], 'Color', 'none');
axis(ax4, 'off');
hold(ax4, 'on');

% Position for color scale labels
x_pos = 30;
text(ax4, x_pos, 256, '+210 KPa', 'Color', 'w', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontWeight', 'bold');
text(ax4, x_pos, 204, '168', 'Color', 'w', 'HorizontalAlignment', 'left');
text(ax4, x_pos, 153, '126', 'Color', 'w', 'HorizontalAlignment', 'left');
text(ax4, x_pos, 102, '84', 'Color', 'w', 'HorizontalAlignment', 'left');
text(ax4, x_pos, 51, '42', 'Color', 'w', 'HorizontalAlignment', 'left');
text(ax4, x_pos, 1, '0', 'Color', 'w', 'HorizontalAlignment', 'left');

% Draw tick marks on colorbar
line(ax4, [0, 5], [256, 256], 'Color', 'w');
line(ax4, [0, 5], [204, 204], 'Color', 'w');
line(ax4, [0, 5], [153, 153], 'Color', 'w');
line(ax4, [0, 5], [102, 102], 'Color', 'w');
line(ax4, [0, 5], [51, 51], 'Color', 'w');
line(ax4, [0, 5], [1, 1], 'Color', 'w');

% Save the result
outputFilename = [files(selectedIdx).name(1:end-4), '_elastography.png'];
outputPath = fullfile(testFolderPath, outputFilename);
saveas(gcf, outputPath);
disp(['Elastography image saved as: ', outputFilename]);

disp('Elastography visualization complete!');

%% Function to perform anisotropic diffusion (noise reduction)
function diffused = anisotropicDiffusion(img, numIter, kappa, lambda, option)
    % Initialize
    diffused = img;
    [rows, cols] = size(img);
    
    % Initialize diffusion coefficient matrices
    c1 = zeros(rows, cols);
    c2 = zeros(rows, cols);
    c3 = zeros(rows, cols);
    c4 = zeros(rows, cols);
    
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
        
        % Apply diffusion update
        diffused = diffused + lambda * (c1.*dN + c2.*dS + c3.*dE + c4.*dW);
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