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
treeCnt_batch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
mtry_batch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
[treeCnt, mtry] = rfValidateParams(X, y, Xval, yval, treeCnt_batch, mtry_batch, 'tempData/rfValidationError.txt', 'tempData/rfMinItem.txt');
lowestCnt = 20; interval = 1;
[treeCnt_batch, mtry_batch] = tuningRandomForest('rfValidationError.txt', 'rfMinItem.txt', lowestCnt, interval);
pause;
% if you want to tune the parameters further, uncomment the following lines
% [treeCnt, mtry] = rfValidateParams(X, y, Xval, yval, treeCnt_batch, mtry_batch, 'tempData/rfValidationError1.txt', 'tempData/rfMinItem1.txt');
% lowestCnt = 20; interval = 1;
% [treeCnt_batch, mtry_batch] = tuningRandomForest('rfValidationError1.txt', 'rfMinItem1.txt', lowestCnt, interval);
% pause;
% [treeCnt, mtry] = rfValidateParams(X, y, Xval, yval, treeCnt_batch, mtry_batch, 'tempData/rfValidationError2.txt', 'tempData/rfMinItem2.txt');
% lowestCnt = 20; interval = 1;
% [treeCnt_batch, mtry_batch] = tuningRandomForest('rfValidationError2.txt', 'rfMinItem2.txt', lowestCnt, interval);
% pause;

%% Training Random Forest
model = regRF_train(X, y, treeCnt, mtry);

%% Prediction
prediction = regRF_predict(Xtest,model);
prediction = (prediction >= 0.5);

accuracy = mean(double(prediction == ytest)) * 100;

fprintf('\nTraining Set Accuracy: %f\n', accuracy);

end

