%% 1. AGID Filtering Analysis
% This script implements and analyzes the Anisotropic Grayscale Intensity Diffusion
% filtering component of the BIDADN framework for breast ultrasound image enhancement

clear all; close all; clc;

%% Setup paths and parameters
testImageFolder = 'test_images/'; % Folder containing test images
outputFolder = 'results/agid/';   % Folder to save AGID results
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% AGID parameters
maxIterations = 50;
k = 0.15;         % Conductance parameter
lambda = 0.25;    % Stabilization parameter

%% Load test images
fprintf('Loading test images...\n');
testImages = dir(fullfile(testImageFolder, '*.png'));

% Ensure at least 10 images are selected
numImages = min(10, length(testImages));  % Use 10 images, or fewer if there aren't 10
sampleImages = cell(numImages, 1);

% Handle case when no images found
if isempty(testImages)
    fprintf('No PNG images found in folder: %s\n', testImageFolder);
    fprintf('Creating simulated test images instead.\n');
    
    % Create simulated ultrasound images
    for i = 1:numImages
        img = zeros(224, 224);
        
        % Add a simulated tumor
        [xx, yy] = meshgrid(1:224, 1:224);
        center_x = 112 + 20*randn();
        center_y = 112 + 20*randn();
        radius = 30 + 10*randn();
        
        circle = ((xx-center_x).^2 + (yy-center_y).^2) < radius^2;
        img(circle) = 0.7 + 0.3*rand(sum(circle(:)), 1);
        
        % Add some speckle noise
        img = img + 0.2*randn(size(img));
        img = mat2gray(img);
        
        % Simulate AGID filtering
        img = agidFiltering(img, maxIterations, k, lambda);
        
        sampleImages{i} = img;
    end
else
    % Select the images to process
    sampleIndices = round(linspace(1, length(testImages), numImages));
    
    for i = 1:numImages
        idx = sampleIndices(i);
        img = imread(fullfile(testImageFolder, testImages(idx).name));
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        sampleImages{i} = im2double(imresize(img, [224, 224]));
    end
end

%% Apply AGID filtering and calculate metrics
filteredImages = cell(size(sampleImages));
snrValues = zeros(length(sampleImages), 2); % Before and after
cnrValues = zeros(length(sampleImages), 2); % Before and after
epiValues = zeros(length(sampleImages), 1);

fprintf('Applying AGID filtering and calculating metrics...\n');

for i = 1:length(sampleImages)
    % Calculate metrics before filtering
    [snrValues(i,1), cnrValues(i,1)] = calculateImageMetrics(sampleImages{i});
    
    % Apply AGID filtering
    tic;
    filteredImages{i} = agidFiltering(sampleImages{i}, maxIterations, k, lambda);
    processingTime = toc;
    
    % Calculate metrics after filtering
    [snrValues(i,2), cnrValues(i,2)] = calculateImageMetrics(filteredImages{i});
    epiValues(i) = calculateEPI(sampleImages{i}, filteredImages{i});
    
    fprintf('Image %d: SNR improved from %.2f dB to %.2f dB (%.1f%%), Processing time: %.3f seconds\n', ...
        i, snrValues(i,1), snrValues(i,2), ...
        (snrValues(i,2)-snrValues(i,1))/snrValues(i,1)*100, processingTime);
end

%% Generate visualizations
fprintf('Generating AGID filtering visualizations...\n');

for i = 1:length(sampleImages)
    % Create detailed filtering visualization
    figure('Position', [100, 100, 1200, 800]);
    
    % Original and filtered images
    subplot(2, 3, 1);
    imshow(sampleImages{i}, []);
    title('Original Ultrasound Image', 'FontWeight', 'bold');
    
    subplot(2, 3, 2);
    imshow(filteredImages{i}, []);
    title('AGID Filtered Image', 'FontWeight', 'bold');
    
    % Edge visualization
    subplot(2, 3, 3);
    [gx1, gy1] = gradient(sampleImages{i});
    [gx2, gy2] = gradient(filteredImages{i});
    gradMag1 = sqrt(gx1.^2 + gy1.^2);
    gradMag2 = sqrt(gx2.^2 + gy2.^2);
    
    edgeCompare = cat(2, mat2gray(gradMag1), mat2gray(gradMag2));
    imshow(edgeCompare, []);
    title('Edge Response: Original (L) vs Filtered (R)', 'FontWeight', 'bold');
    
    % Intensity profile comparison
    subplot(2, 3, 4);
    centerRow = round(size(sampleImages{i}, 1) / 2);
    plot(sampleImages{i}(centerRow, :), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(filteredImages{i}(centerRow, :), 'r-', 'LineWidth', 1.5);
    legend('Original', 'Filtered');
    title('Intensity Profile (Center Row)', 'FontWeight', 'bold');
    xlabel('Column Position');
    ylabel('Intensity');
    grid on;
    
    % Difference image (enhanced for visualization)
    subplot(2, 3, 5);
    diffImg = abs(filteredImages{i} - sampleImages{i});
    imshow(diffImg, []);
    colormap(gca, jet);
    colorbar;
    title('Difference Map (Noise Removed)', 'FontWeight', 'bold');
    
    % Metrics as text
    subplot(2, 3, 6);
    snrImprovement = (snrValues(i,2) - snrValues(i,1)) / snrValues(i,1) * 100;
    cnrImprovement = (cnrValues(i,2) - cnrValues(i,1)) / cnrValues(i,1) * 100;
    
    str = {
        sprintf('SNR before: %.2f dB', snrValues(i,1)),
        sprintf('SNR after: %.2f dB', snrValues(i,2)),
        sprintf('SNR improvement: %.1f%%', snrImprovement),
        sprintf('CNR before: %.2f', cnrValues(i,1)),
        sprintf('CNR after: %.2f', cnrValues(i,2)),
        sprintf('CNR improvement: %.1f%%', cnrImprovement),
        sprintf('Edge Preservation: %.2f', epiValues(i))
    };
    
    axis off;
    text(0.1, 0.5, str, 'FontSize', 12);
    title('Image Quality Metrics', 'FontWeight', 'bold');
    
    % Save figure
    saveas(gcf, fullfile(outputFolder, sprintf('agid_detailed_sample%d.png', i)));
    saveas(gcf, fullfile(outputFolder, sprintf('agid_detailed_sample%d.fig', i)));
    close(gcf);
    
    % Save original and filtered images
    imwrite(im2uint8(mat2gray(sampleImages{i})), fullfile(outputFolder, sprintf('original_sample%d.png', i)));
    imwrite(im2uint8(mat2gray(filteredImages{i})), fullfile(outputFolder, sprintf('filtered_sample%d.png', i)));
end

% Summary and additional charts code can follow here...
