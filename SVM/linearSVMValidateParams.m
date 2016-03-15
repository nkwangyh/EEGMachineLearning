function [C] = linearSVMValidateParams(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Passible values for C and sigma
C_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Train in SVM with the training set X, y
fprintf('Searching for proper params...\n');
fprintf('C    error\n');

m = size(C_batch, 2);

% Save the params and results
linearSVMValidationError = [];

parfor i = 1:m
    C_temp = C_batch(i);
    model= svmTrain(X, y, C_temp, @linearKernel, 1e-3, 20);
    predictions = svmPredict(model, Xval);
    error_temp = mean(double(predictions ~= yval));
        
    linearSVMValidationError = [linearSVMValidationError, [C_temp; error_temp]];
end

[~, minIdx] = min(linearSVMValidationError(2, :));
minItem = linearSVMValidationError(:, minIdx);
C = minItem(1); error = minItem(2);

fprintf('C    error\n');
fprintf(' %f    %f\n', linearSVMValidationError);

fprintf('\nChosen C and error\n  %f  %f\n', C, error);
% =========================================================================
% Save the temporary result as a .mat file to simplify debugging and show
% the primary result in a chart
dlmwrite('linearSVMValidationError.txt', linearSVMValidationError, 'precision', 6, 'delimiter', ' ');
dlmwrite('linearSVMMinItem.txt', minItem, 'precision', 6, 'delimiter', ' ');
% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning

    % for SVM with gaussian kernel, since the two parameters C and sigma are
    % both continous values. Find the lowest 5 columns and get a shrinked range
    % for tuning

    % for random forest, since the two parameters are both discreted values.
    % Find the lowest 5 columns and get a shrinked range for tuning

% the tuning function: take the range as parameters and return a final
% tuning result and illustrate the result on grids

% ========================================================

end
