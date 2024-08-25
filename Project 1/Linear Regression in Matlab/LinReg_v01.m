%% Pre-processing

X_normalized = (X - mean(X)) ./ std(X);
Y_centered = (Y - mean(Y));
%% Variance-Covariance matrix of X

X_cov = (X - mean(X))' * (X - mean(X)) / (size(X,1) - 1) % var-covar matrix X'X/(n-1)
cov(X) % result check
%% Variance-Covariance matrix of X_normalized

X_normalized_cov = X_normalized' * X_normalized / (size(X_normalized,1) - 1) % var-covar matrix
cov(X_normalized) % result check
%% Least Square Regression (X,Y)

w_LS_X = ([ones(size(X,1),1), X]' * [ones(size(X,1),1), X])^(-1) * [ones(size(X,1),1), X]' * Y
Y_LS_X_pred = [ones(size(X,1),1), X] * w_LS_X

%% Least Square Regression (X_normalized,Y_centered)

w_LS = (X_normalized' * X_normalized)^(-1) * X_normalized' * Y_centered
Y_LS_pred = X_normalized * w_LS + mean(Y)

%% Ridge regression(X_normalized,Y_centered)

lambda = 0
w_RR = (lambda * eye(size(X_normalized,2)) + X_normalized' * X_normalized)^(-1) * X_normalized' * Y_centered
Y_RR_pred = X_normalized * w_RR + mean(Y)