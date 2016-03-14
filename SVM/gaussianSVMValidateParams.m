function [C, sigma] = gaussianSVMValidateParams(X, y, Xval, yval)
%GAUSSIANSVMVALIDATEPARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = gaussianSVMValidateParams(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% Passible values for C and sigma
C_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; sigma_batch = C_batch;
% Train in SVM with the training set X, y
fprintf('Searching for proper params...\n');
error = 100;
m = size(C_batch, 2); n = size(sigma_batch, 2);

% Save the params and results
gaussianSVMValidationError = zeros(3, m*n);

for i = 1:m
    C_temp = C_batch(i);
    for j = 1:n
        sigma_temp = sigma_batch(j);
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        error_temp = mean(double(predictions ~= yval)) * 100;
        if error_temp < error
            error = error_temp;
            C = C_temp;
            sigma = sigma_temp;
        end 
        gaussianSVMValidationError(:, (i-1)*n + j) = [C_temp; sigma_temp; error_temp];
    end
end
fprintf('C    sigma    error\n');
fprintf(' %f    %f    %f\n', gaussianSVMValidationError);

fprintf('\nChosen C, sigma and error\n  %f  %f  %f\n', C, sigma, error);
% =========================================================================

end
