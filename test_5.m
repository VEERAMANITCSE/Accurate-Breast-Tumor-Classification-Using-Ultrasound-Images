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


% Handle case when no images found
if isempty(testImages)
    fprintf('No PNG images found in folder: %s\n', testImageFolder);
    fprintf('Creating simulated test images instead.\n');
    
    % Create simulated ultrasound images
    numImages = 5;
    sampleImages = cell(numImages, 1);
    
    
    for i = 1:numImages
        % Create a simulated ultrasound image with speckle noise
        img = zeros(224, 224);
        
        % Add a simulated tumor
        [xx, yy] = meshgrid(1:224, 1:224);
        center_x = 112 + 20*randn();
        center_y = 112 + 20*randn();
        radius = 30 + 10*randn();
        
        % Create circular region with some noise
        circle = ((xx-center_x).^2 + (yy-center_y).^2) < radius^2;
        img(circle) = 0.7 + 0.3*rand(sum(circle(:)), 1);
        
        % Add some speckle noise
        img = img + 0.2*randn(size(img));
        img = mat2gray(img);
        
        sampleImages{i} = img;
    end
else
    % Select sample images
    numImages = min(3, length(testImages));
    sampleIndices = round(linspace(1, length(testImages), numImages));
    sampleImages = cell(numImages, 1);
    
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

% Individual image visualizations
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
    
    % Side-by-side display of edge responses
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

% Generate summary charts
figure('Position', [100, 100, 1200, 400]);

% SNR and CNR improvements
subplot(1, 3, 1);
data = [mean(snrValues(:,1)), mean(snrValues(:,2)); mean(cnrValues(:,1)), mean(cnrValues(:,2))];
b = bar(data, 'grouped');
b(1).FaceColor = [0.3, 0.5, 0.7];
b(2).FaceColor = [0.8, 0.3, 0.3];
set(gca, 'XTickLabel', {'SNR (dB)', 'CNR'});
legend('Before AGID', 'After AGID');
title('Average Image Quality Improvement', 'FontWeight', 'bold');
ylabel('Value');
grid on;

% Add text labels above bars
for i = 1:2
    for j = 1:2
        text(i - 0.15 + (j-1)*0.3, data(i,j) + 0.5, sprintf('%.1f', data(i,j)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

% Improvement percentages
subplot(1, 3, 2);
avg_snr_improvement = (mean(snrValues(:,2)) - mean(snrValues(:,1))) / mean(snrValues(:,1)) * 100;
avg_cnr_improvement = (mean(cnrValues(:,2)) - mean(cnrValues(:,1))) / mean(cnrValues(:,1)) * 100;
avg_epi = mean(epiValues) * 100;

improvement_data = [avg_snr_improvement, avg_cnr_improvement, avg_epi];
b = bar(improvement_data, 'FaceColor', [0.4, 0.7, 0.4]);
set(gca, 'XTickLabel', {'SNR Improvement', 'CNR Improvement', 'Edge Preservation'});
title('AGID Performance Metrics (%)', 'FontWeight', 'bold');
ylabel('Percentage (%)');
ylim([0, max(improvement_data)*1.2]);
grid on;

% Add text labels above bars
for i = 1:length(improvement_data)
    text(i, improvement_data(i) + 2, sprintf('%.1f%%', improvement_data(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Visual comparison of noise profiles
subplot(1, 3, 3);
noiseProfiles = zeros(numImages, 2);
for i = 1:numImages
    % Extract noise estimates by subtracting median filtered image
    origNoise = sampleImages{i} - medfilt2(sampleImages{i}, [5 5]);
    filtNoise = filteredImages{i} - medfilt2(filteredImages{i}, [5 5]);
    
    % Calculate noise standard deviation
    noiseProfiles(i, 1) = std(origNoise(:));
    noiseProfiles(i, 2) = std(filtNoise(:));
end

b = bar(noiseProfiles, 'grouped');
b(1).FaceColor = [0.3, 0.5, 0.7];
b(2).FaceColor = [0.8, 0.3, 0.3];
set(gca, 'XTickLabel', {'Image 1', 'Image 2', 'Image 3'});
legend('Before AGID', 'After AGID');
title('Noise Profile Comparison', 'FontWeight', 'bold');
ylabel('Noise Standard Deviation');
grid on;

% Save the summary figure
saveas(gcf, fullfile(outputFolder, 'agid_summary_metrics.png'));
saveas(gcf, fullfile(outputFolder, 'agid_summary_metrics.fig'));
close(gcf);

%% Helper Functions

function filteredImg = agidFiltering(img, maxIterations, k, lambda)
    % Apply Anisotropic Grayscale Intensity Diffusion filtering
    filteredImg = img;
    
    for t = 1:maxIterations
        % Compute gradients
        [dx, dy] = gradient(filteredImg);
        gradMag = sqrt(dx.^2 + dy.^2);
        
        % Compute diffusion coefficient
        c = exp(-(gradMag.^2)/(k^2));
        
        % Compute divergence (correct implementation)
        cx = c .* dx;
        cy = c .* dy;
        
        [dcx_dx, ~] = gradient(cx);
        [~, dcy_dy] = gradient(cy);
        div = dcx_dx + dcy_dy;
        
        % Update image
        filteredImg = filteredImg + lambda * div;
    end
end

function [snr, cnr] = calculateImageMetrics(img)
    % Calculate Signal-to-Noise Ratio and Contrast-to-Noise Ratio
    
    % Create foreground and background regions
    [rows, cols] = size(img);
    fgMask = false(size(img));
    bgMask = false(size(img));
    
    % Center region for foreground
    centerRow = round(rows/2);
    centerCol = round(cols/2);
    radius = min(rows, cols)/6;
    
    for r = 1:rows
        for c = 1:cols
            dist = sqrt((r-centerRow)^2 + (c-centerCol)^2);
            if dist < radius
                fgMask(r,c) = true;
            elseif dist > radius*2 && dist < radius*3
                bgMask(r,c) = true;
            end
        end
    end
    
    % Calculate metrics
    fgMean = mean(img(fgMask));
    bgMean = mean(img(bgMask));
    bgStd = std(img(bgMask));
    
    % Avoid division by zero
    if bgStd < eps
        bgStd = eps;
    end
    
    % SNR calculation
    snr = 20 * log10(fgMean / bgStd);
    
    % CNR calculation
    cnr = abs(fgMean - bgMean) / bgStd;
end

function epi = calculateEPI(original, filtered)
    % Calculate Edge Preservation Index
    [gx_orig, gy_orig] = gradient(original);
    [gx_filt, gy_filt] = gradient(filtered);
    
    gradMag_orig = sqrt(gx_orig.^2 + gy_orig.^2);
    gradMag_filt = sqrt(gx_filt.^2 + gy_filt.^2);
    
    corr_coeff = corrcoef(gradMag_orig(:), gradMag_filt(:));
    epi = corr_coeff(1,2); % Correlation between original and filtered gradient magnitudes
end
