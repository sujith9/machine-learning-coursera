function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.0;
sigma = 0.1;


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

%{
temp = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
fileID = fopen('exp.txt','w');
for i = 1:length(temp)
    C = temp(i);
    for j = 1:length(temp)
        sigma = temp(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        m = mean(double(predictions ~= yval));
        A1 = [C, sigma, m];
	formatSpec = '%.6f,%.6f,%.6f\n';
	fprintf(fileID,formatSpec,A1);
    end
end
fclose(fileID);
%}



% =========================================================================

end
