function glcmFeatures = extractGLCMFeatures(img)
    % Extract Gray-Level Co-occurrence Matrix (GLCM) features from an image
    % img: Input image (grayscale)
    
    % Compute the GLCM matrix for different offsets
    glcm = graycomatrix(img, 'Offset', [0 1; -1 1; -1 0; -1 -1]);  % 4 different directions
    
    % Extract texture properties from the GLCM
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Extract individual features from the stats structure
    glcmFeatures = [stats.Contrast(:)', stats.Correlation(:)', stats.Energy(:)', stats.Homogeneity(:)'];
    
    % Return the GLCM features as a vector
end
