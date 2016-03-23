function [lambda_batch] = tuningLogisticReg(lrValidationErrorName, minItemName, lowestCnt, paramCnt)
%TUNINGLOGISTICREG show the result of the above coarse-grained validation
%in a chart. Choose 'lowestCnt' groups with lowest error rate and narrow down
% tuning range.
% load data and show the primary result in a chart
lrValidationError = load(lrValidationErrorName);
minItem = load(minItemName);
fprintf('\nChosen lambda and error percent\n  %f  %f\n', minItem(1), 100*minItem(2));
% plot
figure; hold on;
plot(lrValidationError(1, :), lrValidationError(2, :));
miny = min(lrValidationError(2, :)); maxy = max(lrValidationError(2, :)); diffy = maxy - miny;
minx = min(lrValidationError(1, :)); maxx = max(lrValidationError(1, :)); diffx = maxx - minx;
shrinkRate = 20;
if miny ~= maxy
    axis([minx - diffx/shrinkRate, maxx + diffx/shrinkRate, miny - diffy/shrinkRate, maxy + diffy/shrinkRate]);
end
xlabel('lambda'); ylabel('error rate');
plot(minItem(1),  minItem(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning
sortRes = sortrows(lrValidationError', 2);
if lowestCnt > size(sortRes, 1)
    lowestCnt = size(sortRes, 1);
end
lrValidationError_subset = sortRes(1:lowestCnt, :);
minVal = min(lrValidationError_subset(:, 1)); maxVal = max(lrValidationError_subset(:, 1));
% the tuning function: take the range as parameters and return a final
% tuning result and illustrate the result on grids
lambda_batch = linspace(minVal, maxVal, paramCnt);

end

