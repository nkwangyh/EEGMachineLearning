function [ treeCnt, mtry ] = rfValidateParams( X, y, Xval, yval, treeCnt_batch, mtry_batch, rfValidationErrorName, rfMinItemName)
%RFVALIDATEPARAMS adjust the params on cross validation set and return the
%params with highest accuracy

error = 100;
m = size(treeCnt_batch, 2); n = size(mtry_batch, 2);

% Save the params and result
rfValidationError = [];

parfor i = 1:m
    treeCnt_temp = treeCnt_batch(i);
    for j = 1:n
        mtry_temp = mtry_batch(j);
        model = regRF_train(X, y, treeCnt_temp, mtry_temp);
        prediction = regRF_predict(Xval,model);
        prediction = (prediction >= 0.5);
        error_temp = mean(double(prediction ~= yval)) * 100;
        
        rfValidationError = [rfValidationError, [treeCnt_temp; mtry_temp; error_temp]];
    end
end
[~, minIdx] = min(rfValidationError(3, :));
minItem = rfValidationError(:, minIdx);
treeCnt = minItem(1); mtry = minItem(2); error = minItem(3);

fprintf('treeCnt   mtry   error\n');
fprintf('  %f  %f  %f\n', rfValidationError);
fprintf('\nChosen treeCnt, mtry and error percent\n  %f  %f  %f\n', treeCnt, mtry, error);
% =========================================================================

% Save the temporary result as a .mat file to simplify debugging and show
% the primary result in a chart
dlmwrite(rfValidationErrorName, rfValidationError, 'precision', 6, 'delimiter', ' ');
dlmwrite(rfMinItemName, minItem, 'precision', 6, 'delimiter', ' ');
% sort the error matrix using @sortrows and @unique. Return the lowest 5
% columns for detailed tuning

    % for SVM with gaussian kernel, since the two parameters C and sigma are
    % both continous values. Find the lowest 5 columns and get a shrinked range
    % for tuning

    % for random forest, since the two parameters are both discreted values.
    % Find the lowest 5 columns and get a shrinked range for tuning

% the tuning function: take the range as parameters and return a final
% tuning result and illustrate the result on grids

% =========================================================================

end

