function [hidden_layer_size, lambda, maxIter] = nnValidateParams(X, y, Xval, yval, hidden_layer_size, input_layer_size, num_labels)
%NNVALIDATEPARAMS adjust the params on cross validation set and return the
%params with highest accuracy

hidden_layer_size_batch = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28];
lambda_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

maxIter = 100;

options = optimset('MaxIter', maxIter);

m = size(hidden_layer_size_batch, 2); n = size(lambda_batch, 2);
% Save the validation params and result
nnValidationError = [];

parfor i = 1:m
    hidden_layer_size_temp = hidden_layer_size_batch(i);
    for j = 1:n
        lambda_temp = lambda_batch(j);
        initial_nn_params = nnInitializeParameters(hidden_layer_size_temp, input_layer_size, num_labels);
        costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size_temp, ...
                                   num_labels, X, y, lambda_temp);
        [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
        % Obtain Theta1 and Theta2 back from nn_params
        W1 = reshape(nn_params(1:hidden_layer_size_temp*input_layer_size), hidden_layer_size_temp, input_layer_size);
        W2 = reshape(nn_params(hidden_layer_size_temp*input_layer_size+1:hidden_layer_size_temp*input_layer_size+num_labels*hidden_layer_size_temp), num_labels, hidden_layer_size_temp);
        b1 = nn_params(hidden_layer_size_temp*input_layer_size+num_labels*hidden_layer_size_temp+1:hidden_layer_size_temp*input_layer_size+num_labels*hidden_layer_size_temp+hidden_layer_size_temp);
        b2 = nn_params(hidden_layer_size_temp*input_layer_size+num_labels*hidden_layer_size_temp+hidden_layer_size_temp+1:end);
        % Prediction on cross validation set
        pred = nnPredict(W1, W2, b1, b2, Xval);
        error_temp = mean(double(pred' ~= yval)) * 100;
        
        nnValidationError = [nnValidationError, [hidden_layer_size_temp; lambda_temp; error_temp]];
    end
end
[~, minIdx] = min(nnValidationError(3, :));
minItem = nnValidationError(:, minIdx);
hidden_layer_size = minItem(1); lambda = minItem(2); error = minItem(3);
fprintf('hiddenLayerSize  lambda  error\n');
fprintf('%f %f %f\n', nnValidationError);

fprintf('\nChosen hiddenLayerSize, lambda and error percent\n  %f  %f  %f\n', hidden_layer_size, lambda, error);
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

