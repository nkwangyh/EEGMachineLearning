function [accuracy] = neural_network(X, y, Xval, yval, Xtest, ytest)
%NERUAL_NETWORK Train the data with neural network model
%   Assume X is m-by-n input matrix, m is the sample count and n is the
%   demision of the input data; And assume y is a m-by-1 input matrix, each
%   row of y is the label of the same row sample of X. And the function
%   return the training parameters and the accuracy.
%% Load Data and Initialize Parameters
% load data
y(find(y == 0)) = 2;
yval(find(yval == 0)) = 2;
ytest(find(ytest == 0)) = 2;
% initialize parameters
input_layer_size = 14;      % number of input units 
hidden_layer_size = 7;      % number of hidden units 
num_labels = 2;             % output layer
lambda = 0.0001;            % weight decay parameter 
% sparsityParam = 0.01;     % desired average activation of the hidden units.
                            % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                            %  in the lecture notes).
% beta = 3;                 % weight of sparsity penalty term

%% Do some numerical checking


%% Validation
[hidden_layer_size, lambda, maxIter] = nnValidateParams(X, y, Xval, yval, hidden_layer_size, input_layer_size, num_labels);

%% Training NN
fprintf('\nTraining Neural Network... \n');
fprintf('Hidden layer size %f   lambda %f maxIter %f\n', hidden_layer_size, lambda, maxIter);

options = optimset('MaxIter', maxIter);
initial_nn_params = nnInitializeParameters(hidden_layer_size, input_layer_size, num_labels);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                               
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
W1 = reshape(nn_params(1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size);
W2 = reshape(nn_params(hidden_layer_size*input_layer_size+1:hidden_layer_size*input_layer_size+num_labels*hidden_layer_size), num_labels, hidden_layer_size);
b1 = nn_params(hidden_layer_size*input_layer_size+num_labels*hidden_layer_size+1:hidden_layer_size*input_layer_size+num_labels*hidden_layer_size+hidden_layer_size);
b2 = nn_params(hidden_layer_size*input_layer_size+num_labels*hidden_layer_size+hidden_layer_size+1:end);

%% Prediction

pred = nnPredict(W1, W2, b1, b2, Xtest);
accuracy = mean(double(pred' == ytest)) * 100;

fprintf('\nTraining Set Accuracy: %f\n', accuracy);

end

