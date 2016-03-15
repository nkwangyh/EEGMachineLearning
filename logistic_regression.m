function [accuracy, params] = logistic_regression(X, y, Xval, yval, Xtest, ytest)
%LOGISTIC_REGRESSION Train the data with logistic regression model
%   Assume X is m-by-n input matrix, m is the sample count and n is the
%   demision of the input data; And assume y is a m-by-1 input matrix, each
%   row of y is the label of the same row sample of X. And the function
%   return the training parameters and the accuracy.

%% Load Data

% Consider whether to add some polynomial features

% Handle the intercept terms by adding a column of ones for X
m = size(X, 1); mval = size(Xval, 1); mtest = size(Xtest, 1);
X = [ones(m, 1) X]; Xval = [ones(mval, 1) Xval]; Xtest = [ones(mtest, 1) Xtest];

%% Normalize the data if neccessary
% normalize the data with featureNormalize


%% Do some visulization with the training result parameters
% If possible, do some visualization
% plotData(X, y); % hard for high dimension data

%% Validation
% Tune the parameters on the cross-validate set. Iterate the following two functions for several times with
% a coarse-to-fine processing to finally get a fine-tuned result
lambda_batch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
maxIter = 200;
[lambda, maxIter] = LogisticRegValidateParams(X, y, Xval, yval, maxIter, lambda_batch, 'lrValidationError.txt', 'lrMinItem.txt');
lowestCnt = 4; paramCnt = 100;
lambda_batch = tuningLogisticReg('lrValidationError.txt', 'lrMinItem.txt', lowestCnt, paramCnt);
pause;
[lambda, maxIter] = LogisticRegValidateParams(X, y, Xval, yval, maxIter, lambda_batch, 'lrValidationError1.txt', 'lrMinItem1.txt');
lowestCnt = 4; paramCnt = 100;
lambda_batch = tuningLogisticReg('lrValidationError1.txt', 'lrMinItem1.txt', lowestCnt, paramCnt);
pause;

%% Train the model
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set options
options = optimset('GradObj', 'on', 'MaxIter', maxIter);

% Optimize
[theta, J, exit_flag] = ...
    fminunc(@(t)(logisticRegCostFunction(t, X, y, lambda)), initial_theta, options);

%% Predict and compute the accuracy on the test set
% Make predictions w.r.t. the test set and compute the accuray and maybe
% fmeasure result

p = logisticRegPredict(theta, Xtest);
accuracy = mean(double(p == ytest)) * 100;
params = theta;

fprintf('Train Accuracy: %f\n', accuracy);
pause;

end

