function p = nnPredict(W1, W2, b1, b2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(W2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid(W1 * X' + repmat(b1, 1, m));
h2 = sigmoid(W2 * h1 + repmat(b2, 1, m));
[dummy, p] = max(h2, [], 1);

% =========================================================================

end
