function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
all_pairs = zeros(length(C_vec) * length(sigma_vec), 2);
for i = 1 : length(C_vec)
  new_C_vec = ones(length(C_vec), 1) * C_vec(i);
  subset = [new_C_vec sigma_vec];
  all_pairs(i * 8 - 7 : i * 8, :) = [new_C_vec sigma_vec];
endfor
prediction_vec = zeros(size(all_pairs, 1), 1);
for row = 1 : size(all_pairs, 1)
  C = all_pairs(row, 1);
  sigma = all_pairs(row, 2);
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  error = mean(double(svmPredict(model, Xval) ~= yval));
  prediction_vec(row) = error;
endfor
[p, arg_opt] = min(prediction_vec);
C = all_pairs(arg_opt, 1);
sigma = all_pairs(arg_opt, 2);




% =========================================================================

end
