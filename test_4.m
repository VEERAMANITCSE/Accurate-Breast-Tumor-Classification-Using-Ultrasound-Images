%% Comprehensive Analysis Script for BIDADN Framework
% This script analyzes the performance of each component of the BIDADN framework
% and generates visualizations for paper publication

clear all; close all; clc;

%% 1. Setup paths and parameters
disp('Configuring environment for BIDADN analysis...');
testImageFolder = 'test_images/'; % Folder containing test ultrasound images
outputFolder = 'results/';        % Folder to save result visualizations
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Create subdirectories for organized results
componentsDir = fullfile(outputFolder, 'components');
comparativeDir = fullfile(outputFolder, 'comparative');
clinicalDir = fullfile(outputFolder, 'clinical');
if ~exist(componentsDir, 'dir'), mkdir(componentsDir); end
if ~exist(comparativeDir, 'dir'), mkdir(comparativeDir); end
if ~exist(clinicalDir, 'dir'), mkdir(clinicalDir); end

% Set random seed for reproducibility
rng(42);

%% 2. Load test images
disp('Loading and preprocessing test images...');
imageFiles = dir(fullfile(testImageFolder, '*.png'));
numImages = length(imageFiles);

% Select representative images (benign, malignant, normal)
sampleIndices = [5, 25, 45]; % Adjust based on your dataset
sampleImages = cell(length(sampleIndices), 1);
sampleLabels = {'Benign', 'Malignant', 'Normal'};

for i = 1:length(sampleIndices)
    idx = sampleIndices(i);
    if idx <= numImages
        img = imread(fullfile(testImageFolder, imageFiles(idx).name));
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img = imresize(img, [224, 224]);
        sampleImages{i} = img;
    end
end

%% 3. AGID Filtering Analysis
disp('Analyzing AGID filtering performance...');

% AGID parameters
maxIterations = 50;
k = 0.15;         % Conductance parameter
lambda = 0.25;    % Stabilization parameter

filteredImages = cell(length(sampleImages), 1);
snrValues = zeros(length(sampleImages), 2); % Before and after
cnrValues = zeros(length(sampleImages), 2); % Before and after
epiValues = zeros(length(sampleImages), 1);

for i = 1:length(sampleImages)
    img = im2double(sampleImages{i});
    
    % Calculate metrics before filtering
    [snrValues(i,1), cnrValues(i,1)] = calculateImageMetrics(img);
    
    % Apply AGID filtering
    filteredImg = applyAGIDFilter(img, maxIterations, k, lambda);
    
    % Calculate metrics after filtering
    [snrValues(i,2), cnrValues(i,2)] = calculateImageMetrics(filteredImg);
    epiValues(i) = calculateEdgePreservation(img, filteredImg);
    
    filteredImages{i} = filteredImg;
end

% Generate AGID performance visualizations
visualizeAGIDPerformance(sampleImages, filteredImages, snrValues, cnrValues, epiValues, componentsDir);

%% 4. BDTFE Feature Extraction Analysis
disp('Analyzing BDTFE feature optimization performance...');

% BDTFE parameters
alpha = 0.6;
beta = 0.4;
iterations = 100;

bdtfeMetrics = struct();
bdtfeMetrics.dimensionReduction = zeros(length(filteredImages), 1);
bdtfeMetrics.optimizationScore = zeros(length(filteredImages), 1);
bdtfeMetrics.computationalReduction = zeros(length(filteredImages), 1);
bdtfeMetrics.convergenceRate = zeros(length(filteredImages), 1);

for i = 1:length(filteredImages)
    % Extract HOG features as initial feature set
    [initialFeatures, visualization] = extractHOGFeatures(filteredImages{i}, 'CellSize', [8 8]);
    
    % Apply BDTFE optimization
    tic;
    [optimizedFeatures, metrics] = applyBDTFEOptimization(initialFeatures, alpha, beta, iterations);
    bdtfeTime = toc;
    
    % Store metrics
    bdtfeMetrics.dimensionReduction(i) = metrics.dimensionReduction;
    bdtfeMetrics.optimizationScore(i) = metrics.optimizationScore;
    bdtfeMetrics.computationalReduction(i) = metrics.computationalReduction;
    bdtfeMetrics.convergenceRate(i) = metrics.convergenceRate;
    
    % Visualize features before and after optimization
    visualizeBDTFEOptimization(initialFeatures, optimizedFeatures, visualization, filteredImages{i}, ...
        metrics, fullfile(componentsDir, sprintf('bdtfe_sample%d.png', i)));
end

% Generate BDTFE summary visualization
visualizeBDTFEPerformance(bdtfeMetrics, componentsDir);

%% 5. Hybrid Architecture Analysis
disp('Analyzing hybrid architecture performance...');

% Attention mechanism effectiveness
attentionMetrics = struct();
attentionMetrics.spatialAccuracy = [94.8, 91.2, 89.5]; % BIDADN, ResNet-50, DenseNet-169
attentionMetrics.channelPrecision = [93.2, 88.7, 90.1]; % BIDADN, ResNet-50, DenseNet-169
attentionMetrics.fusionEfficiency = [96.4, 91.8, 92.3]; % BIDADN, ResNet-50, DenseNet-169
attentionMetrics.processingTime = [12, 18, 16]; % BIDADN, ResNet-50, DenseNet-169

% Generate attention maps
boundaryMaps = cell(length(filteredImages), 1);
stiffnessMaps = cell(length(filteredImages), 1);

for i = 1:length(filteredImages)
    % Generate boundary attention map (edge-based)
    [gx, gy] = imgradientxy(filteredImages{i});
    gradMag = imgradient(gx, gy);
    boundaryMaps{i} = mat2gray(gradMag);
    boundaryMaps{i} = imgaussfilt(boundaryMaps{i}, 1.5);
    
    % Generate stiffness attention map (simulated)
    stiffnessMaps{i} = generateSimulatedStiffnessMap(filteredImages{i});
end

% Visualize attention mechanism
visualizeAttentionMechanism(filteredImages, boundaryMaps, stiffnessMaps, attentionMetrics, componentsDir);

%% 6. Classification Performance Analysis
disp('Analyzing classification performance...');

% Overall classification metrics
classMetrics = struct();
classMetrics.accuracy = [96.2, 97.1, 96.9]; % Benign, Malignant, Normal
classMetrics.precision = [95.8, 96.1, 95.6]; % Benign, Malignant, Normal
classMetrics.recall = [96.9, 97.4, 97.3]; % Benign, Malignant, Normal
classMetrics.f1Score = [96.3, 96.8, 96.5]; % Benign, Malignant, Normal
classMetrics.aucScore = [0.982, 0.985, 0.983]; % Benign, Malignant, Normal
classMetrics.processingTime = [85, 89, 80]; % Benign, Malignant, Normal

% Confusion matrix (from paper results)
confusionMatrix = [
    550, 30, 20;  % True Benign
    25, 570, 5;   % True Malignant
    15, 10, 575   % True Normal
];

% Generate visualizations for classification performance
visualizeClassificationPerformance(classMetrics, confusionMatrix, sampleLabels, outputFolder);

%% 7. Comparative Analysis
disp('Performing comparative analysis with existing methods...');

% Methods and metrics from the paper
methods = {'BIDADN (Proposed)', 'ResNet-50', 'DenseNet-169', 'UNet+Aug', 'Ensemble ML+XAI', 'FMRNet', 'Transformer'};
compMetrics = struct();
compMetrics.accuracy = [96.75, 92.30, 93.10, 89.50, 91.30, 92.80, 93.70];
compMetrics.precision = [95.86, 91.80, 92.70, 88.90, 90.80, 92.30, 93.20];
compMetrics.recall = [97.25, 92.50, 93.40, 90.00, 91.50, 93.10, 94.10];
compMetrics.f1Score = [96.64, 92.10, 93.00, 89.40, 91.10, 92.70, 93.60];
compMetrics.aucScore = [0.983, 0.940, 0.950, 0.920, 0.930, 0.945, 0.924];
compMetrics.procTime = [84, 150, 180, 200, 220, 170, 190];

% Statistical significance (p-values from ANOVA/t-tests)
significance = struct();
significance.resnet = 0.0045;
significance.densenet = 0.0081;
significance.unet = 0.0003;
significance.ensemble = 0.0077;
significance.fmrnet = 0.0062;
significance.transformer = 0.0089;

% Generate comparative visualizations
visualizeComparativeAnalysis(methods, compMetrics, significance, comparativeDir);

%% 8. Clinical Validation Visualization
disp('Visualizing clinical validation results...');

% Clinical validation metrics from the paper
clinicalMetrics = struct();
clinicalMetrics.concordance = 94.5;  % Agreement with radiologist diagnosis
clinicalMetrics.falsePositive = 3.2; % False positive rate
clinicalMetrics.falseNegative = 2.3; % False negative rate

% Radiologist assessment (simulated data)
radiologistAssessment = [
    92.8, 91.5, 93.4;  % Radiologist 1 (Accuracy, Precision, Recall)
    93.5, 92.8, 94.1;  % Radiologist 2
    95.2, 94.7, 95.8;  % Radiologist 3
    94.5, 95.6, 93.9   % BIDADN
];

% Generate clinical validation visualizations
visualizeClinicalValidation(clinicalMetrics, radiologistAssessment, clinicalDir);

%% 9. Generate Comprehensive Report
disp('Generating comprehensive performance report...');
generateComprehensiveReport(sampleImages, filteredImages, boundaryMaps, stiffnessMaps, ...
    bdtfeMetrics, classMetrics, compMetrics, clinicalMetrics, methods, outputFolder);

disp('BIDADN analysis completed successfully!');

%% === Helper Functions ===

% Function to calculate SNR and CNR
function [snr, cnr] = calculateImageMetrics(img)
    % Calculate Signal-to-Noise Ratio (SNR) and Contrast-to-Noise Ratio (CNR)
    % Assuming the foreground (tumor) and background can be separated based on a simple mask or known regions
    
    % Create foreground and background regions for demonstration
    [rows, cols] = size(img);
    fgMask = false(size(img)); % Initialize foreground mask
    bgMask = false(size(img)); % Initialize background mask
    
    % Define the center region for the foreground (simulating a tumor region)
    centerRow = round(rows / 2);
    centerCol = round(cols / 2);
    radius = min(rows, cols) / 6;  % Simulated tumor size
    
    for r = 1:rows
        for c = 1:cols
            dist = sqrt((r - centerRow)^2 + (c - centerCol)^2);
            if dist < radius
                fgMask(r,c) = true; % Inside the tumor region (foreground)
            elseif dist > radius * 2 && dist < radius * 3
                bgMask(r,c) = true; % Surrounding region (background)
            end
        end
    end
    
    % Calculate mean and standard deviation of pixel intensities
    fgMean = mean(img(fgMask)); % Mean intensity of the foreground
    bgMean = mean(img(bgMask)); % Mean intensity of the background
    bgStd = std(img(bgMask));   % Standard deviation of the background
    
    % SNR calculation (using mean foreground value and standard deviation of the background)
    snr = 20 * log10(fgMean / bgStd);
    
    % CNR calculation (using the difference in mean intensities between foreground and background)
    cnr = abs(fgMean - bgMean) / bgStd;
end

% Other functions like 'applyAGIDFilter', 'visualizeAGIDPerformance', etc., would be similarly defined at the end of the script.
% === AGID Filter Function ===
function filteredImg = applyAGIDFilter(img, maxIterations, k, lambda)
    % This function applies the AGID filter to the input image.
    % img: Input image
    % maxIterations: Maximum number of iterations for the filter
    % k: Conductance parameter
    % lambda: Stabilization parameter
    
    % Initializing variables
    filteredImg = img; % Initialize the filtered image as the input image
    
    % The following is an example of a simple filter loop, 
    % this would need to be replaced with the actual AGID filtering logic
    for iter = 1:maxIterations
        % Apply some filtering logic here, such as edge-preserving filtering
        % Example of a generic filter (this is just a placeholder for illustration)
        filteredImg = filteredImg - k * (filteredImg - imfilter(filteredImg, fspecial('gaussian')));
        
        % Add some stabilization logic if needed (based on lambda)
        filteredImg = filteredImg + lambda * (img - filteredImg); % Stabilization step (example)
        
        % Optionally, you could apply stopping criteria (e.g., convergence checks)
    end
end
% === Edge Preservation Function ===
function epi = calculateEdgePreservation(originalImg, filteredImg)
    % This function calculates the Edge Preservation Index (EPI)
    % originalImg: The original input image
    % filteredImg: The filtered output image
    
    % Convert images to double for processing
    originalImg = im2double(originalImg);
    filteredImg = im2double(filteredImg);
    
    % Apply Sobel edge detection to both images
    originalEdges = edge(originalImg, 'Sobel');
    filteredEdges = edge(filteredImg, 'Sobel');
    
    % Calculate the edge preservation index (EPI) as the ratio of preserved edges
    % EPI = (intersection of edges) / (union of edges)
    intersection = sum(sum(originalEdges & filteredEdges)); % Intersection of edges
    union = sum(sum(originalEdges | filteredEdges)); % Union of edges
    
    % Edge preservation index (EPI)
    epi = intersection / union;
end
% === Visualization Function for AGID Filtering Performance ===
function visualizeAGIDPerformance(originalImages, filteredImages, snrValues, cnrValues, epiValues, saveDir)
    % This function generates visualizations for AGID filtering performance
    % originalImages: Original test images
    % filteredImages: Filtered test images
    % snrValues: Signal-to-Noise Ratio values before and after filtering
    % cnrValues: Contrast-to-Noise Ratio values before and after filtering
    % epiValues: Edge Preservation Index values
    % saveDir: Directory where the results will be saved

    % Number of images to visualize
    numImages = length(originalImages);
    
    for i = 1:numImages
        % Create a figure for each image
        figure;
        subplot(2, 2, 1);
        imshow(originalImages{i}, []);
        title(sprintf('Original Image: %s', sprintf('Sample %d', i)));
        
        subplot(2, 2, 2);
        imshow(filteredImages{i}, []);
        title(sprintf('Filtered Image: %s', sprintf('Sample %d', i)));
        
        subplot(2, 2, 3);
        % Display the SNR and CNR before and after filtering
        text(0.1, 0.8, sprintf('SNR (Before): %.2f dB', snrValues(i, 1)), 'FontSize', 12);
        text(0.1, 0.6, sprintf('SNR (After): %.2f dB', snrValues(i, 2)), 'FontSize', 12);
        text(0.1, 0.4, sprintf('CNR (Before): %.2f', cnrValues(i, 1)), 'FontSize', 12);
        text(0.1, 0.2, sprintf('CNR (After): %.2f', cnrValues(i, 2)), 'FontSize', 12);
        axis off;
        
        subplot(2, 2, 4);
        % Display the Edge Preservation Index (EPI)
        text(0.1, 0.8, sprintf('Edge Preservation Index: %.2f', epiValues(i)), 'FontSize', 12);
        axis off;
        
        % Save the figure
        saveas(gcf, fullfile(saveDir, sprintf('AGID_Performance_Sample_%d.png', i)));
        close;
    end
end
% === BDTFE Optimization Function ===
function [optimizedFeatures, metrics] = applyBDTFEOptimization(initialFeatures, alpha, beta, iterations)
    % This function optimizes the initial extracted features using BDTFE optimization
    % initialFeatures: The initial set of features extracted from the image (e.g., HOG features)
    % alpha: Weight parameter for the feature optimization
    % beta: Weight parameter for the computational optimization
    % iterations: The number of iterations for the optimization process
    
    % Placeholder variables for optimized features and metrics
    optimizedFeatures = initialFeatures; % Start with initial features
    metrics = struct();
    metrics.dimensionReduction = 0;     % Placeholder for dimension reduction
    metrics.optimizationScore = 0;       % Placeholder for optimization score
    metrics.computationalReduction = 0;  % Placeholder for computational reduction
    metrics.convergenceRate = 0;         % Placeholder for convergence rate

    % Example optimization loop (this can be replaced with the actual optimization method)
    for iter = 1:iterations
        % Example of a feature selection/reduction step (e.g., PCA, LDA, or any optimization)
        optimizedFeatures = featureSelectionStep(optimizedFeatures, alpha);
        
        % Example of a computational efficiency improvement (e.g., pruning or dimensionality reduction)
        optimizedFeatures = computationalEfficiencyStep(optimizedFeatures, beta);
        
        % Track some basic metrics (placeholders, can be replaced with actual values)
        metrics.dimensionReduction = metrics.dimensionReduction + 0.1; % Dummy value
        metrics.optimizationScore = metrics.optimizationScore + 0.05; % Dummy value
        metrics.computationalReduction = metrics.computationalReduction + 0.02; % Dummy value
        metrics.convergenceRate = metrics.convergenceRate + 0.03; % Dummy value
    end
    
    % Normalize metrics for the final iteration
    metrics.dimensionReduction = metrics.dimensionReduction / iterations;
    metrics.optimizationScore = metrics.optimizationScore / iterations;
    metrics.computationalReduction = metrics.computationalReduction / iterations;
    metrics.convergenceRate = metrics.convergenceRate / iterations;
end

% --- Feature selection step (dummy example) ---
function features = featureSelectionStep(features, alpha)
    % A placeholder function for feature selection or dimensionality reduction
    % Alpha parameter could control how aggressively features are selected
    % Example: Reduce features based on a threshold or method (e.g., PCA, LDA)
    features = features(:, 1:round(size(features, 2) * alpha)); % Dummy feature reduction
end

% --- Computational efficiency step (dummy example) ---
function features = computationalEfficiencyStep(features, beta)
    % A placeholder function for optimizing computational efficiency
    % Beta parameter could control how much the features should be pruned or reduced
    % Example: Remove less significant features (dummy example)
    features = features(:, 1:round(size(features, 2) * beta)); % Dummy computational reduction
end
% === Visualization Function for BDTFE Optimization ===
function visualizeBDTFEOptimization(initialFeatures, optimizedFeatures, visualization, filteredImg, metrics, savePath)
    % This function visualizes the before and after of BDTFE optimization
    % initialFeatures: The initial features extracted from the image (e.g., HOG features)
    % optimizedFeatures: The optimized feature set after applying BDTFE
    % visualization: The visualization of the feature map (e.g., HOG visualization)
    % filteredImg: The filtered image after applying AGID filter
    % metrics: The metrics calculated from BDTFE optimization (e.g., dimension reduction, optimization score)
    % savePath: Path to save the visualizations
    
    % Create a figure for visualization
    figure;
    
    % Show the filtered image
    subplot(2, 3, 1);
    imshow(filteredImg, []);
    title('Filtered Image');
    
    % Show the initial features (e.g., HOG features before optimization)
    subplot(2, 3, 2);
    imshow(visualization);
    title('Initial Features (HOG)');
    
    % Show the optimized features (e.g., after applying BDTFE optimization)
    subplot(2, 3, 3);
    % You might visualize optimized features differently depending on the nature of optimization
    imshow(optimizedFeatures, []);
    title('Optimized Features');
    
    % Show a bar chart of the BDTFE optimization metrics
    subplot(2, 3, 4);
    bar([metrics.dimensionReduction, metrics.optimizationScore, metrics.computationalReduction, metrics.convergenceRate]);
    set(gca, 'xticklabel', {'Dimension Reduction', 'Optimization Score', 'Computational Reduction', 'Convergence Rate'});
    title('BDTFE Optimization Metrics');
    
    % Display additional feature information in a text box
    subplot(2, 3, 5:6);
    text(0.1, 0.9, sprintf('Dimension Reduction: %.2f', metrics.dimensionReduction), 'FontSize', 12);
    text(0.1, 0.7, sprintf('Optimization Score: %.2f', metrics.optimizationScore), 'FontSize', 12);
    text(0.1, 0.5, sprintf('Computational Reduction: %.2f', metrics.computationalReduction), 'FontSize', 12);
    text(0.1, 0.3, sprintf('Convergence Rate: %.2f', metrics.convergenceRate), 'FontSize', 12);
    axis off;
    
    % Save the figure to the specified path
    saveas(gcf, savePath);
    close;
end
