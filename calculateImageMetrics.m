function [snr, cnr] = calculateImageMetrics(img)
    % Calculate Signal-to-Noise Ratio (SNR) and Contrast-to-Noise Ratio (CNR)
    % img: Input image (grayscale)
    
    % Create foreground and background regions for the purpose of calculating metrics
    [rows, cols] = size(img);
    fgMask = false(size(img));  % Initialize foreground mask
    bgMask = false(size(img));  % Initialize background mask
    
    % Define a center region for the foreground (simulating a tumor region)
    centerRow = round(rows / 2);
    centerCol = round(cols / 2);
    radius = min(rows, cols) / 6;  % Simulated tumor size
    
    % Assign pixels to foreground and background based on distance from the center
    for r = 1:rows
        for c = 1:cols
            dist = sqrt((r - centerRow)^2 + (c - centerCol)^2);
            if dist < radius
                fgMask(r,c) = true;  % Inside the tumor region (foreground)
            elseif dist > radius * 2 && dist < radius * 3
                bgMask(r,c) = true;  % Surrounding region (background)
            end
        end
    end
    
    % Calculate mean and standard deviation of pixel intensities for foreground and background
    fgMean = mean(img(fgMask));  % Mean intensity of the foreground
    bgMean = mean(img(bgMask));  % Mean intensity of the background
    bgStd = std(img(bgMask));    % Standard deviation of the background
    
    % SNR calculation (using mean foreground value and standard deviation of the background)
    if bgStd > 0
        snr = 20 * log10(fgMean / bgStd);  % Convert ratio to decibels
    else
        snr = Inf;  % If background standard deviation is zero, return infinite SNR
    end
    
    % CNR calculation (using the difference in mean intensities between foreground and background)
    if bgStd > 0
        cnr = abs(fgMean - bgMean) / bgStd;  % Contrast normalized by background noise
    else
        cnr = Inf;  % If background standard deviation is zero, return infinite CNR
    end
end
