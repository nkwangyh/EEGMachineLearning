function [lambda, maxIter] = LogisticRegValidateParams(X, y, Xval, yval)
%LOGISTICREGVALIDATEPARAMS Train the model on train set with different params
% and test the result on cross validation set to return the best params.

lambda_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

maxIter = 200;
options = optimset('GradObj', 'on', 'MaxIter', maxIter);

n = size(X, 2);
initial_theta = zeros(n, 1);
m = size(lambda_batch, 2);

% Save the validation params and result
lrValidationError = [];

parfor i = 1:m
    lambda_temp = lambda_batch(i);
    % Tune 'MaxIter' accroding to lambda, which means MaxIter should
    % increase with lambda's increasing
    [theta, ~, ~] = ...
    fminunc(@(t)(logisticRegCostFunction(t, X, y, lambda_temp)), initial_theta, options);
    predictions = logisticRegPredict(theta, Xval);
    error_temp = mean(double(predictions ~= yval));
    
    lrValidationError = [lrValidationError, [lambda_temp; error_temp]];
end
[~, minIdx] = min(lrValidationError(2, :));
minItem = lrValidationError(:, minIdx);
lambda = minItem(1); error = minItem(2);

fprintf('Logistic Regression: \n  lambda  error\n');
fprintf('  %f  %f\n', lrValidationError);
fprintf('\nChosen lambda, error percent and maxIter\n  %f  %f  %f\n', lambda, error, maxIter);

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

end

