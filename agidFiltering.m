% AGID Filtering Function: Anisotropic Grayscale Intensity Diffusion Filtering
function filteredImg = agidFiltering(img, maxIterations, k, lambda)
    % Apply Anisotropic Grayscale Intensity Diffusion filtering
    % img: Input image (grayscale)
    % maxIterations: Maximum number of iterations
    % k: Conductance parameter
    % lambda: Stabilization parameter (controls the degree of diffusion)

    filteredImg = img;  % Initialize the filtered image as the original image
    
    for t = 1:maxIterations
        % Compute gradients (partial derivatives) in both directions
        [dx, dy] = gradient(filteredImg);
        
        % Calculate gradient magnitude (edge strength)
        gradMag = sqrt(dx.^2 + dy.^2);
        
        % Compute diffusion coefficient based on the gradient magnitude
        c = exp(-(gradMag.^2) / (k^2));
        
        % Compute the directional components of the diffusion
        cx = c .* dx;
        cy = c .* dy;
        
        % Compute the divergence (rate of change) of the diffusion
        [dcx_dx, ~] = gradient(cx);
        [~, dcy_dy] = gradient(cy);
        
        % Calculate the image update (diffusion term)
        div = dcx_dx + dcy_dy;
        
        % Update the image based on the computed divergence
        filteredImg = filteredImg + lambda * div;
    end
end
