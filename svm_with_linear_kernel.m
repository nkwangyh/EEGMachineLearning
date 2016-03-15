function [accuracy, params] = svm_with_linear_kernel(X, y, Xval, yval, Xtest, ytest)
%SVM Train the data with neural SVM model
%   Assume X is m-by-n input matrix, m is the sample count and n is the
%   demision of the input data; And assume y is a m-by-1 input matrix, each
%   row of y is the label of the same row sample of X. And the function
%   return the training parameters and the accuracy.

%% Training Linear SVM
fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)

%% Validation
% Try different SVM Parameters here
C = linearSVMValidateParams(X, y, Xval, yval);

%% Predict
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
p = svmPredict(model, Xtest);

params = C;
accuracy = mean(double(p == ytest)) * 100;

fprintf('Training Accuracy: %f\n', accuracy);

end

