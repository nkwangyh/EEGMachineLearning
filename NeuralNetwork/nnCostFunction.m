function [cost grad] = nnCostFunction(nn_params, ...
                                   visibleSize, ...
                                   hiddenSize, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
W1 = reshape(nn_params(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(nn_params(hiddenSize*visibleSize+1:hiddenSize*visibleSize+num_labels*hiddenSize), num_labels, hiddenSize);
b1 = nn_params(hiddenSize*visibleSize+num_labels*hiddenSize+1:hiddenSize*visibleSize+num_labels*hiddenSize+hiddenSize);
b2 = nn_params(hiddenSize*visibleSize+num_labels*hiddenSize+hiddenSize+1:end);

% Setup some useful variables
m = size(X, 1);
% Transpose the data, make one column a sample
data = X';
% Vectorize label y
vecY = eye(num_labels);
Y = zeros(num_labels, m);
for i = 1:size(y, 1)
    Y(:, i) = vecY(:, y(i));
end

% feedforward
b1_batch = repmat(b1, 1, m); b2_batch = repmat(b2, 1, m);
z_2 = W1 * data + b1_batch; a_2 = sigmoid(z_2);
z_3 = W2 * a_2 + b2_batch; a_3 = sigmoid(z_3);
% measuring cost by squaring the difference
%cost = 1/(2*m) * sum(sum((a_3 - Y) .^ 2));
%reg_cost = lambda/2 * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));

% measuring cost by logrithm the difference
cost = -1/m * (sum(sum(Y .* log(a_3) + (ones(size(Y)) - Y) .* log(ones(size(a_3)) - a_3))));
reg_cost = lambda/(2 * m) * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
cost = cost + reg_cost;

% backpropagation
% delta_3 = -(Y - a_3) .* a_3 .* (1 - a_3);
delta_3 = -(Y - a_3);
delta_2 = W2' * delta_3 .* a_2 .* (1 - a_2);
delta_W2 = delta_3 * a_2'; delta_W1 = delta_2 * data';

W2grad = 1/m .* delta_W2 + lambda/m .* W2; W1grad = 1/m .* delta_W1 + lambda/m .* W1;
b2grad = 1/m .* sum(delta_3, 2); b1grad = 1/m .* sum(delta_2, 2);

% =========================================================================

% Unroll gradients
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
