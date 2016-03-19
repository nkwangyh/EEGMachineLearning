function [ treeCnt_batch, mtry_batch ] = tuningRandomForest( rfValidationErrorName, rfMinItemName, lowestCnt, interval )
%TUNINGRANDOMFOREST show the result of the above coarse-grained validation
%in a chart. Choose 'lowestCnt' groups with lowest error rate and narrow down
% tuning range.

% load data and show the primary result in a chart
rfValidationError = load(rfValidationErrorName);
minItem = load(rfMinItemName);
fprintf('\nChosen treeCnt, mtry and error percent\n  %f  %f  %f\n', minItem(1), minItem(2), minItem(3));
% handling data
treeCnt = unique(rfValidationError(1, :), 'stable'); mtry = unique(rfValidationError(2, :), 'stable');
errorMatrix = reshape(rfValidationError(3, :), length(mtry), length(treeCnt));

% bar3 plot
figure;
bar3(errorMatrix);
xlabel('treeCnt'); ylabel('mtry');

% Surface plot
figure;
surf(treeCnt, mtry, errorMatrix);
xlabel('treeCnt'); ylabel('mtry');

% Contour plot
figure;
contour(treeCnt, mtry, errorMatrix);
xlabel('treeCnt'); ylabel('mtry');
hold on;
plot(minItem(1), minItem(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning
sortRes = sortrows(rfValidationError', 3);
if lowestCnt > size(sortRes, 1)
    lowestCnt = size(sortRes, 1);
end
rfValidationError_subset = sortRes(1:lowestCnt, :);
minTreeCnt = min(rfValidationError_subset(:, 1)); maxTreeCnt = max(rfValidationError_subset(:, 1));
minMtry = min(rfValidationError_subset(:, 2)); maxMtry = max(rfValidationError_subset(:, 2));

treeCnt_batch = minTreeCnt:interval:maxTreeCnt;
mtry_batch = minMtry:interval:maxMtry;

end

