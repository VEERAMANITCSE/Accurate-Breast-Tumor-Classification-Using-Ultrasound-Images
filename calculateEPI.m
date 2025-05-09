function epi = calculateEPI(original, filtered)
    % Calculate Edge Preservation Index (EPI)
    % original: The original image
    % filtered: The filtered image
    
    % Compute gradients (partial derivatives) for both original and filtered images
    [gx_orig, gy_orig] = gradient(original);
    [gx_filt, gy_filt] = gradient(filtered);
    
    % Calculate gradient magnitudes for both images
    gradMag_orig = sqrt(gx_orig.^2 + gy_orig.^2);
    gradMag_filt = sqrt(gx_filt.^2 + gy_filt.^2);
    
    % Compute the correlation coefficient between original and filtered gradient magnitudes
    corr_coeff = corrcoef(gradMag_orig(:), gradMag_filt(:));
    
    % EPI is the correlation between the gradients of the original and filtered images
    epi = corr_coeff(1,2);  % Extract correlation coefficient
end
