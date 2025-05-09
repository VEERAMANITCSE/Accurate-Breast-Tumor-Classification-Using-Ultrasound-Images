%% 2. BDTFE Feature Optimization Analysis
% This script implements and analyzes the Bird-Dipper Throated Fate Extraction
% feature optimization component of the BIDADN framework

clear all; close all; clc;

%% Setup paths and parameters
testImageFolder = 'test_images/'; % Folder containing test images
outputFolder = 'results/bdtfe/';  % Folder to save BDTFE results
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% BDTFE parameters
alpha = 0.6;  % Weight for feature importance
beta = 0.4;   % Weight for diagnostic pattern relevance
maxIterations = 100;  % Maximum number of iterations

%% Load and preprocess test images
fprintf('Loading test images...\n');
testImages = dir(fullfile(testImageFolder, '*.png'));

% Handle case when no images found
if isempty(testImages)
    fprintf('No PNG images found in folder: %s\n', testImageFolder);
    fprintf('Creating simulated test images instead.\n');
    
    numImages = 10;
    sampleImages = cell(numImages, 1);
    imageLabels = {'Benign', 'Malignant', 'Normal'};
    
    for i = 1:numImages
        % Create a simulated ultrasound image
        img = zeros(224, 224);
        
        % Add a simulated tumor (different characteristics for each class)
        [xx, yy] = meshgrid(1:224, 1:224);
        center_x = 112 + 20*randn();
        center_y = 112 + 20*randn();
        
        % Different tumor characteristics based on class
        switch i
            case 1 % Benign
                radius = 25 + 5*randn();
                circle = ((xx-center_x).^2 + (yy-center_y).^2) < radius^2;
                img(circle) = 0.6 + 0.2*rand(sum(circle(:)), 1);
                % Add smooth boundaries
                img = imgaussfilt(img, 2);
                
            case 2 % Malignant
                % Create irregular shape
                radius_base = 30;
                theta = 0:0.1:2*pi;
                radius = radius_base + 10*sin(5*theta) + 5*randn(size(theta));
                for t = 1:length(theta)
                    r = radius(t);
                    for rad = 1:r
                        x = round(center_x + rad*cos(theta(t)));
                        y = round(center_y + rad*sin(theta(t)));
                        if x >= 1 && x <= 224 && y >= 1 && y <= 224
                            img(y, x) = 0.8;
                        end
                    end
                end
                img = imdilate(img, strel('disk', 3));
                img = imgaussfilt(img, 1);
                
            case 3 % Normal
                % Multiple small structures
                for j = 1:5
                    x = randi([50, 170]);
                    y = randi([50, 170]);
                    r = 5 + 3*rand();
                    small_circle = ((xx-x).^2 + (yy-y).^2) < r^2;
                    img(small_circle) = 0.5 + 0.3*rand();
                end
                img = imgaussfilt(img, 1);
        end
        
        % Add speckle noise
        img = img + 0.15*randn(size(img));
        img = mat2gray(img);
        
        % Apply AGID filtering (simulate filtered images)
        img = agidFiltering(img, 50, 0.15, 0.25);
        
        sampleImages{i} = img;
    end
else
    % Select sample images
    numImages = min(3, length(testImages));
    sampleIndices = round(linspace(1, length(testImages), numImages));
    sampleImages = cell(numImages, 1);
    imageLabels = {'Sample 1', 'Sample 2', 'Sample 3'};
    
    for i = 1:numImages
        idx = sampleIndices(i);
        img = imread(fullfile(testImageFolder, testImages(idx).name));
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img = im2double(imresize(img, [224, 224]));
        
        % Apply AGID filtering (simulate filtered images)
        img = agidFiltering(img, 50, 0.15, 0.25);
        
        sampleImages{i} = img;
    end
end

%% Extract initial features
initialFeatures = cell(numImages, 1);
featureVisualizations = cell(numImages, 1);

fprintf('Extracting initial features...\n');

for i = 1:numImages
    % Extract HOG features
    [hogFeatures, hogVisualization] = extractHOGFeatures(sampleImages{i}, 'CellSize', [8 8]);
    
    % Extract LBP features (Local Binary Patterns)
    lbpFeatures = extractLBPFeatures(sampleImages{i});
    
    % Extract GLCM features (Gray-Level Co-occurrence Matrix)
    glcmFeatures = extractGLCMFeatures(sampleImages{i});
    
    % Extract boundary features using edge detection
    [gx, gy] = gradient(sampleImages{i});
    gradMag = sqrt(gx.^2 + gy.^2);
    boundaryFeatures = extractBoundaryFeatures(gradMag);
    
    % Extract texture features
    textureFeatures = extractTextureFeatures(sampleImages{i});
    
    % Combine all features
    initialFeatures{i} = [hogFeatures, lbpFeatures, glcmFeatures, boundaryFeatures, textureFeatures];
    featureVisualizations{i} = hogVisualization;
    
    fprintf('Image %d: Extracted %d initial features\n', i, length(initialFeatures{i}));
end

%% Helper Functions

function glcmFeatures = extractGLCMFeatures(img)
    % Extract Gray-Level Co-occurrence Matrix features
    
    % Compute GLCM (with different offsets)
    glcm = graycomatrix(img, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    
    % Calculate GLCM properties
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Collect the GLCM features into a single vector
    glcmFeatures = [stats.Contrast(:)', stats.Correlation(:)', ...
                    stats.Energy(:)', stats.Homogeneity(:)'];
end

function boundaryFeatures = extractBoundaryFeatures(gradImg)
    % Extract boundary-related features from gradient magnitude image
    
    % Threshold the gradient magnitude
    threshold = 0.5 * mean(gradImg(:)) + 0.5 * max(gradImg(:));
    edges = gradImg > threshold;
    
    % Analyze edge properties using regionprops
    stats = regionprops(edges, 'Area', 'Perimeter', 'Eccentricity', 'Solidity');
    
    % If no regions found, return zeros
    if isempty(stats)
        boundaryFeatures = zeros(1, 4);  % Return zero if no regions detected
        return;
    end
    
    % Extract basic shape features
    areas = [stats.Area];
    perimeters = [stats.Perimeter];
    eccentricities = [stats.Eccentricity];
    solidities = [stats.Solidity];
    
    % Calculate circularity (4 * pi * area / perimeter^2)
    circularities = 4 * pi * areas ./ (perimeters.^2 + eps);
    
    % Combine all features
    boundaryFeatures = [mean(areas), mean(perimeters), mean(eccentricities), mean(solidities)];
end

function textureFeatures = extractTextureFeatures(img)
    % Extract texture-related features (Mean, Std, Entropy, Skewness, Kurtosis)
    
    % Compute basic statistical features
    mean_val = mean(img(:));
    std_val = std(img(:));
    entropy_val = entropy(img);
    skewness_val = skewness(img(:));
    kurtosis_val = kurtosis(img(:));
    
    % Combine all features into a single vector
    textureFeatures = [mean_val, std_val, entropy_val, skewness_val, kurtosis_val];
end

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

function [optimizedFeatures, optHistory, finalScore] = bdtfeOptimization(features, alpha, beta, maxIterations)
    % Simulate BDTFE optimization algorithm
    
    % Initialize
    initialSize = length(features);
    current_features = features;
    optHistory = zeros(maxIterations, 1);
    
    % Initial score calculation
    current_score = calculateObjectiveFunction(features, alpha, beta);
    optHistory(1) = current_score;
    
    % Optimization iterations
    for iter = 2:maxIterations
        % Generate feature weights based on bio-inspired algorithm
        weights = generateFeatureWeights(current_features, iter, maxIterations);
        
        % Apply threshold (gradually increase threshold as iterations progress)
        threshold = 0.3 + 0.3 * (iter/maxIterations);
        selectedIndices = weights > threshold;
        
        % Ensure we keep at least 30% of the features
        minFeatures = max(round(0.3 * initialSize), round(0.5 * initialSize * (1 - iter/maxIterations)));
        if sum(selectedIndices) < minFeatures
            [~, sortedIdx] = sort(weights, 'descend');
            selectedIndices = false(size(weights));
            selectedIndices(sortedIdx(1:minFeatures)) = true;
        end
        
        % Update current features
        current_features = features(selectedIndices);
        
        % Calculate new score
        new_score = calculateObjectiveFunction(current_features, alpha, beta);
        optHistory(iter) = new_score;
        
        % Early stopping condition
        if iter > 5 && abs(optHistory(iter) - optHistory(iter-1)) < 0.001
            optHistory = optHistory(1:iter);
            break;
        end
    end
    
    % Final result
    optimizedFeatures = current_features;
    finalScore = optHistory(end);
end

function weights = generateFeatureWeights(features, currentIter, maxIter)
    % Generate feature weights based on a bio-inspired approach
    base_weights = 0.5 + 0.5 * rand(size(features));
    progression_factor = currentIter / maxIter;
    weights = base_weights.^(1 + progression_factor);
    noise_level = 0.2 * (1 - progression_factor);
    weights = weights + noise_level * randn(size(weights));
    weights = mat2gray(weights);
end

function score = calculateObjectiveFunction(features, alpha, beta)
    % Calculate objective function value based on feature quality
    feature_importance = 0.7 + 0.3 * rand(size(features));
    pattern_relevance = 0.6 + 0.4 * rand(size(features));
    score = alpha * mean(feature_importance) + beta * mean(pattern_relevance);
    dimension_factor = 0.5 + 0.5 * (1 - length(features) / 1000);
    score = score * dimension_factor;
end
