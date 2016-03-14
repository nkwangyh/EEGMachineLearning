function [accuracy, params] = svm_with_gaussian_kernel(X, y, Xval, yval, Xtest, ytest)
%SVM Train the data with neural SVM model
%   Assume X is m-by-n input matrix, m is the sample count and n is the
%   demision of the input data; And assume y is a m-by-1 input matrix, each
%   row of y is the label of the same row sample of X. And the function
%   return the training parameters and the accuracy.

%% Training SVM with RBF Kernel
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% SVM Parameters
% C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

%% Validation
% Try different SVM Parameters here
[C, sigma] = gaussianSVMValidateParams(X, y, Xval, yval);

%% Predict
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
p = svmPredict(model, Xtest);

params = [C; sigma];
accuracy = mean(double(p == ytest)) * 100;

fprintf('Training Accuracy: %f\n', accuracy);

end

