function [hidden_layer_size_batch, lambda_batch] = tuningNeuralNetwork(nnValidationErrorName, nnMinItemName, lowestCnt, paramCnt)
%TUNINGNEURALNETWORK show the result of the above coarse-grained validation
%in a chart. Choose 'lowestCnt' groups with lowest error rate and narrow down
% tuning range.
% load data and show the primary result in a chart
nnValidationError = load(nnValidationErrorName);
minItem = load(nnMinItemName);
%fprintf('\nChosen hiddenLayerSize, lambda and error percent\n  %f  %f  %f\n', minItem(1), minItem(2), minItem(3));
% handling data
hiddenSize = unique(nnValidationError(1, :), 'stable'); lambda = unique(nnValidationError(2, :), 'stable');
errorMatrix = reshape(nnValidationError(3, :), length(lambda), length(hiddenSize));
% bar3 plot
figure;
bar3(errorMatrix);
xlabel('hidden layer size'); ylabel('lambda');

% Surface plot
figure;
surf(hiddenSize, lambda, errorMatrix);
xlabel('hidden layer size'); ylabel('lambda');

% Contour plot
figure;
contour(hiddenSize, lambda, errorMatrix);
xlabel('hidden layer size'); ylabel('lambda');
hold on;
plot(minItem(1), minItem(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning
sortRes = sortrows(nnValidationError', 3);
if lowestCnt > size(sortRes, 1)
    lowestCnt = size(sortRes, 1);
end
nnValidationError_subset = sortRes(1:lowestCnt, :);
minHiddenSize = min(nnValidationError_subset(:, 1)); maxHiddenSize = max(nnValidationError_subset(:, 1));
minLambda = min(nnValidationError_subset(:, 2)); maxLambda = max(nnValidationError_subset(:, 2));

hidden_layer_size_batch = minHiddenSize:1:maxHiddenSize;
lambda_batch = linspace(minLambda, maxLambda, paramCnt);

end

