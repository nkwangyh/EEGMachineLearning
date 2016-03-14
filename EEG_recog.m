%% Project for EEG recoginition with Machine Learning

%% Initialization
clear; close all; clc

addpath('LogisticRegression/');
addpath('NeuralNetwork/');
addpath('SVM/');
addpath('RandomForest/');
addpath('data/');

%% Load Data and Divide it into Train, Cross-validation and Test sets
fprintf('Loading data ...\n');
% Load positive data and add label to it

% pos_data = load('data_1.txt'); pos_data = pos_data(:, 2:15);
pos_data = load('data_2_1.txt'); pos_data_temp = load('data_2_2.txt'); pos_data = [pos_data; pos_data_temp];
pos_data = pos_data(:, 2:15);
% If there is another data, load the data and incorporate it to pos_data
% label positive samples as 1
pos_data = [pos_data ones(size(pos_data, 1), 1)];
% Load negtive data and add label to it

% neg_data = load('data_0.txt'); neg_data = neg_data(:, 2:15);
neg_data = load('data_8_1.txt'); neg_data_temp = load('data_8_2.txt'); neg_data = [neg_data; neg_data_temp];
neg_data = neg_data(:, 2:15);
% If there is another data, load the data and incorporate it to pos_data
% label negtive samples as 0
neg_data = [neg_data zeros(size(neg_data, 1), 1)];
% randomly pick 1000 data from pos_data and neg_data respectively
cnt = 10000; % change if necessary
m1 = size(pos_data, 1); m2 = size(neg_data, 1);
if (m1 < cnt) || (m2 < cnt)
    fprintf('there is not enough data samples\n');
    pause;
end
randposidx = randperm(m1); randnegidx = randperm(m2);
pos_data = pos_data(randposidx(1:cnt), :);
neg_data = neg_data(randnegidx(1:cnt), :);

data = [pos_data; neg_data];

% Reorgnize the data randomly
% Randomly reorder the indices of examples
m = size(data, 1); n = size(data, 2) - 1;
randidx = randperm(m);
randdata = data(randidx, :);
train_set_size = floor(0.6 * m); val_set_size = floor(0.2 * m); test_set_size = m - train_set_size - val_set_size;

% Divide the data into three data sets
X = randdata(1:train_set_size, 1:n); 
y = randdata(1:train_set_size, n+1);
Xval= randdata(train_set_size+1: train_set_size+val_set_size, 1:n);
yval = randdata(train_set_size+1: train_set_size+val_set_size, n+1);
Xtest = randdata(train_set_size+val_set_size+1:m, 1:n);
ytest = randdata(train_set_size+val_set_size+1:m, n+1);

fprintf('Loading complete.\n');
% disp(size(X)); disp(size(y));
% disp(size(Xval));disp(size(yval));
% disp(size(Xtest));disp(size(ytest));

%% Train with logistic regression
fprintf('Training with logistic regression ...\n');
logistic_reg_res = logistic_regression(X, y, Xval, yval, Xtest, ytest);
fprintf('Logistic regression training result: %f\n', logistic_reg_res);
pause;

%% Train with Neural Network
% fprintf('Training with Neural Network ...\n');
% nn_res = neural_network(X, y, Xval, yval, Xtest, ytest);
% fprintf('Neural network training result: %f\n', nn_res);
% pause;

%% Train with SVM with linear kernel or Gaussian kernel
% fprintf('Training with SVM with linear kernel ...\n');
% svm_linear_res = svm_with_linear_kernel(X, y, Xval, yval, Xtest, ytest);
% fprintf('SVM with linear kernel training result: %f\n', svm_linear_res);
% pause;

% fprintf('Training with SVM with gaussian kernel ...\n');
% svm_gaussian_res = svm_with_gaussian_kernel(X, y, Xval, yval, Xtest, ytest);
% fprintf('SVM with gaussian kernel training result: %f\n', svm_gaussian_res);
% pause;

%% Train with Random Forest
% fprintf('Training with Random Forest ...\n');
% rf_res = random_forest(X, y, Xval, yval, Xtest, ytest);
% fprintf('Random forest training result: %f\n', rf_res);
% pause;

%% Compare the results of the several method and get a conclusion