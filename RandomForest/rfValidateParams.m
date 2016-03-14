function [ treeCnt, mtry ] = rfValidateParams( X, y, Xval, yval )
%RFVALIDATEPARAMS adjust the params on cross validation set and return the
%params with highest accuracy
treeCnt_batch = [10, 50, 100];
mtry_batch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

error = 100;
m = size(treeCnt_batch, 2); n = size(mtry_batch, 2);
for i = 1:m
    treeCnt_temp = treeCnt_batch(i);
    for j = 1:n
        mtry_temp = mtry_batch(j);
        model = regRF_train(X, y, treeCnt_temp, mtry_temp);
        prediction = regRF_predict(Xval,model);
        prediction = (prediction >= 0.5);
        error_temp = mean(double(prediction ~= yval)) * 100;
        
        if error_temp < error
            error = error_temp;
            treeCnt = treeCnt_temp;
            mtry = mtry_temp;
        end
        fprintf('treeCnt   mtry   error\n %f  %f  %f\n', treeCnt_temp, mtry_temp, error_temp);
    end
end

end

