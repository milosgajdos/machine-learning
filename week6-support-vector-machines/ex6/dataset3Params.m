function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

param_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
% error_mx stores prediction error for each combination of C and sigma
% row indexes represent C, column indexes represent sigma
error_mx  = zeros(length(param_vec), length(param_vec));

for C = 1:length(param_vec)
    for sigma = 1:length(param_vec)
        % compute SVM model parameters
        model = svmTrain(X, y, param_vec(C), ...
            @(x1, x2) gaussianKernel(x1, x2, param_vec(sigma)));
        % compute prediction for Cross Validation set
        predictions = svmPredict(model, Xval);
        % compute prediction error and store in error matrix
        error_mx(C,sigma) = mean(double(predictions ~= yval));
    end
end

% find minimum index in the whole error matrix
% I contains minimum prediction error found in error_mx matrix
[~,I] = min(error_mx(:));
% C_best and sigma_best contain indexes for best C and sigma
[C_best_idx, sigma_best_idx] = ind2sub(size(error_mx),I);
% lookup best C and sigma using C_best_dx, sigma_best_idx
C = param_vec(C_best_idx);
sigma = param_vec(sigma_best_idx);

% =========================================================================

end
