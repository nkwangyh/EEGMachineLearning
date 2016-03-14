function [C, sigma] = gaussianSVMValidateParams(X, y, Xval, yval)
%GAUSSIANSVMVALIDATEPARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = gaussianSVMValidateParams(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Passible values for C and sigma
C_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; sigma_batch = C_batch;
% Train in SVM with the training set X, y
fprintf('Searching for proper params...\n');

m = size(C_batch, 2); n = size(sigma_batch, 2);

% Save the params and results
gaussianSVMValidationError = [];

parfor i = 1:m
    C_temp = C_batch(i);
    for j = 1:n
        sigma_temp = sigma_batch(j);
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        error_temp = mean(double(predictions ~= yval)) * 100;
        
        gaussianSVMValidationError = [gaussianSVMValidationError, [C_temp; sigma_temp; error_temp]];
    end
end
[~, minIdx] = min(gaussianSVMValidationError(3, :));
minItem = gaussianSVMValidationError(:, minIdx);
C = minItem(1); sigma = minItem(2); error = minItem(3);

fprintf('C    sigma    error\n');
fprintf(' %f    %f    %f\n', gaussianSVMValidationError);

fprintf('\nChosen C, sigma and error\n  %f  %f  %f\n', C, sigma, error);
% =========================================================================

% Save the temporary result as a .mat file to simplify debugging and show
% the primary result in a chart

% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning

    % for SVM with gaussian kernel, since the two parameters C and sigma are
    % both continous values. Find the lowest 5 columns and get a shrinked range
    % for tuning

    % for random forest, since the two parameters are both discreted values.
    % Find the lowest 5 columns and get a shrinked range for tuning

% the tuning function: take the range as parameters and return a final
% tuning result and illustrate the result on grids

% =========================================================================
end
