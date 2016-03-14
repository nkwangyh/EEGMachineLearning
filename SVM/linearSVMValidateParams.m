function [C, sigma] = linearSVMValidateParams(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Passible values for C and sigma
C_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; sigma_batch = C_batch;
% Train in SVM with the training set X, y
fprintf('Searching for proper params...\n');
fprintf('C    sigma    error\n');
error = 1;
m = size(C_batch, 2); n = size(sigma_batch, 2);
for i = 1:m
    C_temp = C_batch(i);
    for j = 1:n
        sigma_temp = sigma_batch(j);
        model= svmTrain(X, y, C, @linearKernel, 1e-3, 20);
        predictions = svmPredict(model, Xval);
        error_temp = mean(double(predictions ~= yval));
        fprintf(' %f    %f    %f\n', C_temp, sigma_temp, error_temp);
        if error_temp < error
            error = error_temp;
            C = C_temp;
            sigma = sigma_temp;
        end        
    end
end
% =========================================================================

end