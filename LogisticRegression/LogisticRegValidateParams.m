function [lambda, maxIter] = LogisticRegValidateParams(X, y, Xval, yval)
%LOGISTICREGVALIDATEPARAMS Train the model on train set with different params
% and test the result on cross validation set to return the best params.

lambda = 0.01;
lambda_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

maxIter = 500;
options = optimset('GradObj', 'on', 'MaxIter', maxIter);

error = 1;
n = size(X, 2);
initial_theta = zeros(n, 1);
m = size(lambda_batch, 2);
fprintf('Logistic Regression: \n        lambda          error\n');
for i = 1:m
    lambda_temp = lambda_batch(i);
    % Tune 'MaxIter' accroding to lambda, which means MaxIter should
    % increase with lambda's increasing
    [theta, ~, ~] = ...
    fminunc(@(t)(logisticRegCostFunction(t, X, y, lambda_temp)), initial_theta, options);
    predictions = logisticRegPredict(theta, Xval);
    error_temp = mean(double(predictions ~= yval));
    
    if error_temp < error
        lambda = lambda_temp;
        error = error_temp;
    end
    fprintf('        %f          %f\n', lambda_temp, error_temp);
end

end

