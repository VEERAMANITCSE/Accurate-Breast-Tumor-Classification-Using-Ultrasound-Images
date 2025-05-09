function textureFeatures = extractTextureFeatures(img)
    % Extract texture-related features from an image
    % img: Input image (grayscale)
    
    % Basic statistical features
    mean_val = mean(img(:));
    std_val = std(img(:));
    entropy_val = entropy(img);
    skewness_val = skewness(img(:));
    kurtosis_val = kurtosis(img(:));
    
    % Tamura texture features (simplified for this example)
    % Coarseness
    coarseness = 1 / (1 + std_val);  % Simplified version
    
    % Contrast (measured by standard deviation of pixel intensities)
    contrast = std_val / (mean_val^0.25);
    
    % Directionality (simplified)
    [gx, gy] = gradient(img);
    gradient_direction = atan2(gy, gx);
    directionality = std(gradient_direction(:));
    
    % Combine all features
    textureFeatures = [mean_val, std_val, entropy_val, skewness_val, kurtosis_val, ...
                       coarseness, contrast, directionality];
    
    % Optionally add more texture features as needed
    textureFeatures = [textureFeatures, rand(1, 12)]; % Simulated extra features for illustration
end
