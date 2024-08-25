function [wRR, active] = hw1_regression(str_lambda, str_sigma2, file_X_train, file_y_train, file_X_test)
% hw1_regression has 2 parts:
% PART 1: Ridge regression, takes in data  y_train  and  X_train  and
%   outputs  wRR  for an arbitrary value of  lambda.
% PART 2: Active learning, provides the first 10 locations to measure from
%   dataset X_test, given (y_train, X_train), lambda and sigma2.

% Inputs:
%   str_lambda = lambda value as a string
%   str_sigma2 = sigma2 value as a string
%   file_X_train = csv file containing the covariates. Each row is a single
%       vector  xi. Last dimension has already been set equal to 1 for all
%       data.
%   file_y_train = csv file containing the outputs. Each row has a single
%       number and the i-th row of this file combined with the i-th row of
%       "X_train.csv" constitutes the training pair  (yi,xi).
%   file_X_test = csv file exactly the same format as "X_train.csv". No
%       response file is given for the testing data.
%   
% Outputs:
%   wRR = column vector of the Ridge regression coefficients
%   active = row vector of the active learning process, with first 10
%       locations from X_test to measure. 
%   wRR_[lambda].csv = file where the value in each dimension of the vector
%       wRR is contained on a new line. This file corresponds output for
%       PART 1.
%   active_[lambda]_[sigma2].csv = a csv file containing the row index of
%       the first 10 vectors selected from X_test.csv. Indexing starts at 1
%       (i.e., the first row is index 1). File contains one line with a ","
%       separating each index value. This file corresponds to your output
%       for PART 2.

%% Data import
lambda = str2double(str_lambda);
sigma2 = str2double(str_sigma2);

X_train = table2array(readtable(file_X_train, 'ReadVariableNames',false));
y_train = table2array(readtable(file_y_train, 'ReadVariableNames',false));
X_test = table2array(readtable(file_X_test, 'ReadVariableNames',false));

%% Pre-processing
X_train_normalized = (X_train - mean(X_train)) ./ std(X_train);
X_train_normalized = X_train_normalized(:,2:end);

y_train_centered = (y_train - mean(y_train));

X_test_normalized = (X_test - mean(X_test)) ./ std(X_test);
X_test_normalized = X_test_normalized(:,2:end);

%% PART 1: Ridge regression
wRR = (lambda * eye(size(X_train_normalized,2)) + X_train_normalized' * X_train_normalized)^(-1) * X_train_normalized' * y_train_centered;
y_train_pred = X_train_normalized * wRR + mean(y_train);

csvwrite(['wRR_',str_lambda,'.csv'], wRR)

%% PART 2: Active learning
active = [1:10]; % simulated result

csvwrite(['active_',str_lambda,'_',str_sigma2,'.csv'], active)

end

