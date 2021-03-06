function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET6PARAMS returns your choice of C and sigma for Part 3 of the exercise
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

% Create vectors of C and sigma values

C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

C_opt = 0;
sigma_opt = 0;

model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

predictions = svmPredict(model,Xval);

error_temp = mean(double(predictions ~= yval));

% For loops to go over C and sigma

for i = 1:size(C_vect)
    for j = 1:size(sigma_vect)
        C = C_vect(i);
        sigma = sigma_vect(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        predictions = svmPredict(model,Xval);
        
        error = mean(double(predictions ~= yval));
        
        if(error<error_temp)
            error_temp = error;
            C_opt = C;
            sigma_opt = sigma;
        end
    end
end

C = C_opt;
sigma = sigma_opt;



% =========================================================================

end
