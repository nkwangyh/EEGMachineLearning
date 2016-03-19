function [ C_batch, sigma_batch ] = tuningGaussianKernelSVM( gaussianSVMValidationErrorName, gaussianSVMMinItemName, lowestCnt, param_C_cnt, param_sigma_cnt )
%TUNINGGAUSSIANKERNELSVM show the result of the above coarse-grained validation
%in a chart. Choose 'lowestCnt' groups with lowest error rate and narrow down
% tuning range.
% load data and show the primary result in a chart
gaussianSVMValidationError = load(gaussianSVMValidationErrorName);
minItem = load(gaussianSVMMinItemName);
fprintf('\nChosen C, sigma and error\n  %f  %f  %f\n', minItem(1), minItem(2), minItem(3));
% handling data
C = unique(gaussianSVMValidationError(1, :), 'stable'); sigma = unique(gaussianSVMValidationError(2, :), 'stable');
errorMatrix = reshape(gaussianSVMValidationError(3, :), length(sigma), length(C));

% bar3 plot
figure;
bar3(errorMatrix);
xlabel('C'); ylabel('sigma');

% Surface plot
figure;
surf(C, sigma, errorMatrix);
xlabel('C'); ylabel('sigma');

% Contour plot
figure;
contour(C, sigma, errorMatrix);
xlabel('C'); ylabel('sigma');
hold on;
plot(minItem(1), minItem(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning
sortRes = sortrows(gaussianSVMValidationError', 3);
if lowestCnt > size(sortRes, 1)
    lowestCnt = size(sortRes, 1);
end
gaussianSVMValidationError_subset = sortRes(1:lowestCnt, :);
minC = min(gaussianSVMValidationError_subset(:, 1)); maxC = max(gaussianSVMValidationError_subset(:, 1));
minSigma = min(gaussianSVMValidationError_subset(:, 2)); maxSigma = max(gaussianSVMValidationError_subset(:, 2));

C_batch = linspace(minC, maxC, param_C_cnt);
sigma_batch = linspace(minSigma, maxSigma, param_sigma_cnt);

end

