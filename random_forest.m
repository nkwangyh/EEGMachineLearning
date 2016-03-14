function [ accuracy ] = random_forest(X, y, Xval, yval, Xtest, ytest)
%RANDOM_FOREST Train the data with random forest model
%   Assume X is m-by-n input matrix, m is the sample count and n is the
%   demision of the input data; And assume y is a m-by-1 input matrix, each
%   row of y is the label of the same row sample of X. And the function
%   return the training parameters and the accuracy.

%% Load Data and Initialize Parameters
%compile everything
%once compiled, the code for compiling can be commented

% if strcmpi(computer,'PCWIN') || strcmpi(computer,'PCWIN64')
%    compile_windows
% else
%    compile_linux
% end

%% Validation
[treeCnt, mtry] = rfValidateParams(X, y, Xval, yval);

%% Training Random Forest
model = regRF_train(X, y, treeCnt, mtry);

%% Prediction
prediction = regRF_predict(Xtest,model);
prediction = (prediction >= 0.5);

accuracy = mean(double(prediction == ytest)) * 100;

fprintf('\nTraining Set Accuracy: %f\n', accuracy);

end

