function boundaryFeatures = extractBoundaryFeatures(gradImg)
    % Extract boundary-related features from a gradient magnitude image
    % gradImg: Gradient magnitude image
    
    % Threshold the gradient magnitude to get the edges
    threshold = 0.5 * mean(gradImg(:)) + 0.5 * max(gradImg(:)); % Adaptive threshold
    edges = gradImg > threshold;  % Edge detection based on thresholding
    
    % Use regionprops to measure properties of the detected boundaries
    stats = regionprops(edges, 'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'BoundingBox');
    
    % If no boundaries are found, return zeros
    if isempty(stats)
        boundaryFeatures = zeros(1, 20);
        return;
    end
    
    % Extract basic shape features
    areas = [stats.Area];
    perimeters = [stats.Perimeter];
    eccentricities = [stats.Eccentricity];
    solidities = [stats.Solidity];
    
    % Calculate circularity (4?*area/perimeter^2)
    circularities = 4 * pi * areas ./ (perimeters.^2 + eps);
    
    % Combine the features into a vector
    boundaryFeatures = [
        mean(areas), std(areas), ...              % Area statistics
        mean(perimeters), std(perimeters), ...    % Perimeter statistics
        mean(eccentricities), std(eccentricities), ... % Eccentricity statistics
        mean(solidities), std(solidities), ...    % Solidity statistics
        mean(circularities), std(circularities)   % Circularity statistics
    ];
    
    % Optionally, you can also include other statistical features like
    % gradient information (mean and std of gradImg) or other region-based metrics
    gradStats = [mean(gradImg(:)), std(gradImg(:)), max(gradImg(:)), ...
                 prctile(gradImg(:), 75), median(gradImg(:)), ...
                 prctile(gradImg(:), 25), min(gradImg(:)), skewness(gradImg(:)), ...
                 kurtosis(gradImg(:)), entropy(gradImg(:))];
    
    % Append gradient statistics to the boundary features
    boundaryFeatures = [boundaryFeatures, gradStats];
end
