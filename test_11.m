%% BDTFE Feature Optimization Analysis
% Bird-Dipper Throated Fate Extraction for the BIDADN framework

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

% Handle case when no images are found
if isempty(testImages)
    fprintf('No PNG images found in folder: %s\n', testImageFolder);
    fprintf('Creating simulated test images instead.\n');
    numImages = 3;
    sampleImages = cell(numImages, 1);
    imageLabels = {'Benign', 'Malignant', 'Normal'};
    
    for i = 1:numImages
        img = zeros(224, 224);
        [xx, yy] = meshgrid(1:224, 1:224);
        center_x = 112 + 20*randn();
        center_y = 112 + 20*randn();
        
        % Different tumor characteristics based on class
        switch i
            case 1 % Benign
                radius = 25 + 5*randn();
                circle = ((xx-center_x).^2 + (yy-center_y).^2) < radius^2;
                img(circle) = 0.6 + 0.2*rand(sum(circle(:)), 1);
                img = imgaussfilt(img, 2);
                
            case 2 % Malignant
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
                for j = 1:5
                    x = randi([50, 170]);
                    y = randi([50, 170]);
                    r = 5 + 3*rand();
                    small_circle = ((xx-x).^2 + (yy-y).^2) < r^2;
                    img(small_circle) = 0.5 + 0.3*rand();
                end
                img = imgaussfilt(img, 1);
        end
        
        img = img + 0.15*randn(size(img));
        img = mat2gray(img);
        sampleImages{i} = img;
    end
else
    numImages = min(3, length(testImages));
    sampleImages = cell(numImages, 1);
    for i = 1:numImages
        img = imread(fullfile(testImageFolder, testImages(i).name));
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        sampleImages{i} = im2double(imresize(img, [224, 224]));
    end
end

%% Feature Extraction
fprintf('Extracting initial features...\n');
initialFeatures = cell(numImages, 1);
for i = 1:numImages
    [hogFeatures, hogVisualization] = extractHOGFeatures(sampleImages{i}, 'CellSize', [8 8]);
    lbpFeatures = extractLBPFeatures(sampleImages{i});
    glcmFeatures = extractGLCMFeatures(sampleImages{i});
    gradMag = imgradient(sampleImages{i});
    textureFeatures = stdfilt(sampleImages{i});
    initialFeatures{i} = [hogFeatures, lbpFeatures, glcmFeatures, gradMag(:)', textureFeatures(:)'];
    fprintf('Image %d: Extracted %d initial features\n', i, length(initialFeatures{i}));
end

%% BDTFE Optimization
fprintf('Applying BDTFE optimization...\n');
optimizedFeatures = cell(numImages, 1);
for i = 1:numImages
    [optimizedFeatures{i}, ~, finalScore] = bdtfeOptimization(initialFeatures{i}, alpha, beta, maxIterations);
    fprintf('Image %d: Optimization score: %.3f\n', i, finalScore);
end

%% Visualization
fprintf('Generating visualizations...\n');
for i = 1:numImages
    figure;
    subplot(1, 2, 1);
    imshow(sampleImages{i}, []);
    title(sprintf('Sample Image %d', i));
    
    subplot(1, 2, 2);
    plot(hogVisualization);
    title('HOG Features Visualization');
    saveas(gcf, fullfile(outputFolder, sprintf('bdtfe_image_%d.png', i)));
    % close(gcf);
end

fprintf('BDTFE optimization analysis completed.\n');
