function [ J, grad ] = logisticRegCostFunction( theta, X, y, lambda )
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

theta_x = X*theta;
prediction = sigmoid(theta_x);
diff = prediction - y;
J = -1/m * sum(y .* log(prediction) + (1 - y) .* log(1 - prediction)) ...
     + lambda / (2 * m) * sum(theta .^ 2);

% compute gradient
grad = 1/m .* (diff' * X)' - lambda/m * theta;
grad(1) = grad(1) + lambda/m * theta(1);

end

