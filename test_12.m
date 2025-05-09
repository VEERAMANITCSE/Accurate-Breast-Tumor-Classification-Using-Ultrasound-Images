%% BIDADN Classification Analysis with Confusion Matrices
% This script analyzes the classification performance of the BIDADN framework
% with a focus on how boundary and tissue stiffness features contribute to the
% classification outcomes.

clear all; close all; clc;

%% Setup paths
outputFolder = 'results/classification/';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Set random seed for reproducibility
rng(42);

%% 1. Define ground truth and simulation parameters
% Real classification results would come from the actual BIDADN model
% Here we simulate results based on performance reported in the paper

% Number of samples per class
numSamples = 600;

% Class labels
classLabels = {'Benign', 'Malignant', 'Normal'};
numClasses = length(classLabels);

% Accuracy metrics from the paper
overallAccuracy = 0.9675;
classAccuracy = [0.962, 0.971, 0.969]; % Benign, Malignant, Normal
precision = [0.958, 0.961, 0.956];
recall = [0.969, 0.974, 0.973];
f1Score = [0.963, 0.968, 0.965];

% Ground truth (true classes)
groundTruth = [];
for i = 1:numClasses
    groundTruth = [groundTruth; repmat(i, numSamples, 1)];
end

% Total number of samples
totalSamples = length(groundTruth);

%% 2. Simulate classification results based on reported metrics
% Create a confusion matrix based on reported performance

% Base confusion matrix from paper
baseConfusionMatrix = [
    550, 30, 20;  % True Benign predictions
    25, 570, 5;   % True Malignant predictions
    15, 10, 575   % True Normal predictions
];

% Create predicted labels based on the confusion matrix
predictedLabels = zeros(size(groundTruth));
currentIndex = 1;

for trueClass = 1:numClasses
    for predClass = 1:numClasses
        numPredictions = baseConfusionMatrix(trueClass, predClass);
        indices = currentIndex:(currentIndex + numPredictions - 1);
        predictedLabels(indices) = predClass;
        currentIndex = currentIndex + numPredictions;
    end
end

% Verify total samples
assert(currentIndex - 1 == totalSamples, 'Sample count mismatch');

%% 3. Analyze classification performance with different configurations

% Configuration 1: Overall model (BIDADN with both branches)
fprintf('Analyzing overall BIDADN performance...\n');
analyzeBIDADNPerformance(groundTruth, predictedLabels, classLabels, 'Overall BIDADN', outputFolder);

% Simulate results with only boundary features (ResNet50 branch)
% Typically would have slightly worse performance, especially for texture-dependent cases
boundaryOnlyPerformance = 0.91; % Simulated based on ablation studies
boundaryPredictions = simulateReducedPerformance(groundTruth, predictedLabels, boundaryOnlyPerformance);
fprintf('Analyzing performance with only boundary features (ResNet50 branch)...\n');
analyzeBIDADNPerformance(groundTruth, boundaryPredictions, classLabels, 'Boundary Features Only', outputFolder);

% Simulate results with only stiffness features (DenseNet169 branch)
stiffnessOnlyPerformance = 0.89; % Simulated based on ablation studies
stiffnessPredictions = simulateReducedPerformance(groundTruth, predictedLabels, stiffnessOnlyPerformance);
fprintf('Analyzing performance with only stiffness features (DenseNet169 branch)...\n');
analyzeBIDADNPerformance(groundTruth, stiffnessPredictions, classLabels, 'Stiffness Features Only', outputFolder);

% Create a subset of challenging cases where boundary and stiffness features are complementary
challengingCases = identifyChallengingCases(groundTruth, boundaryPredictions, stiffnessPredictions, predictedLabels);
fprintf('Analyzing complementary effects on challenging cases...\n');
analyzeComplementaryEffects(challengingCases, classLabels, outputFolder);

%% 4. Perform comparative analysis of feature contributions
fprintf('Performing comparative analysis of feature contributions...\n');
compareFeatureContributions(groundTruth, predictedLabels, boundaryPredictions, stiffnessPredictions, classLabels, outputFolder);

%% 5. Generate summary visualization
fprintf('Generating summary visualization...\n');
generateSummaryVisualization(groundTruth, predictedLabels, boundaryPredictions, stiffnessPredictions, classLabels, outputFolder);

fprintf('Classification analysis completed. Results saved to %s\n', outputFolder);

%% Helper Functions

function analyzeBIDADNPerformance(groundTruth, predictions, classLabels, modelName, outputFolder)
    % Calculate confusion matrix
    numClasses = length(classLabels);
    cm = zeros(numClasses, numClasses);
    
    for i = 1:length(groundTruth)
        trueClass = groundTruth(i);
        predClass = predictions(i);
        cm(trueClass, predClass) = cm(trueClass, predClass) + 1;
    end
    
    % Calculate metrics
    accuracy = sum(diag(cm)) / sum(cm(:));
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    
    for i = 1:numClasses
        precision(i) = cm(i,i) / sum(cm(:,i));
        recall(i) = cm(i,i) / sum(cm(i,:));
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
    
    % Plot confusion matrix
    figure('Position', [100, 100, 800, 600]);
    
    % Create the heatmap
    imagesc(cm);
    colormap('hot');
    colorbar;
    
    % Add labels
    title(sprintf('%s Confusion Matrix (Accuracy: %.2f%%)', modelName, accuracy*100), 'FontWeight', 'bold');
    xlabel('Predicted Class');
    ylabel('True Class');
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classLabels);
    set(gca, 'YTick', 1:numClasses, 'YTickLabel', classLabels);
    
    % Add text annotations for each cell
    for i = 1:numClasses
        for j = 1:numClasses
            text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
                'Color', [1 1 1]);
        end
    end
    
    % Save figure
    saveas(gcf, fullfile(outputFolder, sprintf('confusion_matrix_%s.png', strrep(modelName, ' ', '_'))));
    saveas(gcf, fullfile(outputFolder, sprintf('confusion_matrix_%s.fig', strrep(modelName, ' ', '_'))));
    close(gcf);
    
    % Plot metrics
    figure('Position', [100, 100, 900, 400]);
    
    % Performance metrics per class
    subplot(1, 2, 1);
    metrics = [precision, recall, f1];
    bar(metrics);
    set(gca, 'XTickLabel', classLabels);
    ylabel('Value');
    legend('Precision', 'Recall', 'F1-Score');
    title(sprintf('%s Performance Metrics by Class', modelName), 'FontWeight', 'bold');
    grid on;
    
    % Normalized confusion matrix (percentage)
    subplot(1, 2, 2);
    cm_norm = cm ./ sum(cm, 2);
    imagesc(cm_norm);
    colormap('hot');
    colorbar;
    
    title('Normalized Confusion Matrix (%)', 'FontWeight', 'bold');
    xlabel('Predicted Class');
    ylabel('True Class');
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classLabels);
    set(gca, 'YTick', 1:numClasses, 'YTickLabel', classLabels);
    
    % Add percentage text
    for i = 1:numClasses
        for j = 1:numClasses
            text(j, i, sprintf('%.1f%%', cm_norm(i,j)*100), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
                'Color', [1 1 1]);
        end
    end
    
    % Save figure
    saveas(gcf, fullfile(outputFolder, sprintf('metrics_%s.png', strrep(modelName, ' ', '_'))));
    saveas(gcf, fullfile(outputFolder, sprintf('metrics_%s.fig', strrep(modelName, ' ', '_'))));
    close(gcf);
    
    % Print metrics to console
    fprintf('\n%s Performance Metrics:\n', modelName);
    fprintf('Overall Accuracy: %.2f%%\n', accuracy*100);
    fprintf('Class-wise Metrics:\n');
    
    for i = 1:numClasses
        fprintf('%s - Precision: %.2f%%, Recall: %.2f%%, F1-Score: %.2f%%\n', ...
            classLabels{i}, precision(i)*100, recall(i)*100, f1(i)*100);
    end
    fprintf('\n');
end

function reducedPredictions = simulateReducedPerformance(groundTruth, fullPredictions, targetAccuracy)
    % Simulates reduced performance by introducing errors to match target accuracy
    % This function maintains the error distribution pattern while reducing overall accuracy
    
    currentAccuracy = mean(groundTruth == fullPredictions);
    errorRate = 1 - currentAccuracy;
    targetErrorRate = 1 - targetAccuracy;
    
    % Scale factor to increase errors
    scaleFactor = targetErrorRate / errorRate;
    
    % Initialize with full predictions
    reducedPredictions = fullPredictions;
    
    % Identify correct predictions
    correctIndices = find(groundTruth == fullPredictions);
    
    % Calculate number of correct predictions to convert to errors
    numToConvert = round((currentAccuracy - targetAccuracy) * length(groundTruth));
    
    % Randomly select correct predictions to make incorrect
    if numToConvert > 0
        indicesToConvert = correctIndices(randperm(length(correctIndices), numToConvert));
        
        % For each selected index, change the prediction to an incorrect class
        for i = 1:length(indicesToConvert)
            idx = indicesToConvert(i);
            trueClass = groundTruth(idx);
            
            % Select a random incorrect class
            possibleClasses = setdiff(unique(groundTruth), trueClass);
            newClass = possibleClasses(randi(length(possibleClasses)));
            
            % Assign the incorrect class
            reducedPredictions(idx) = newClass;
        end
    end
end
