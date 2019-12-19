function [err_train, err_test] = ModelSpherical(train, test)
%You implment this function by assuming a spherical covariance matrix. 
%err_train is the error rate on the train data
%err_test is the error rate on the test data

[~, d] = size(train);

xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

xtrain = double(xtrain./255);
xtest = double(xtest ./255);

pos_idx = find(ytrain == 1);
neg_idx = find(ytrain == 0);

pi_pos = length(pos_idx);
pi_neg = length(neg_idx);

pos_mean = mean(xtrain(pos_idx, :));
neg_mean = mean(xtrain(neg_idx, :));

pos_var = mean(diag(cov(xtrain(pos_idx, :))));
neg_var = mean(diag(cov(xtrain(neg_idx, :))));

epsilon = 1e-2;

V_pos = 1./sqrt(pos_var + epsilon) * eye(d-1);
V_neg = 1./sqrt(neg_var + epsilon) * eye(d-1);
S_pos = repmat(pos_var, d-1, 1);
S_neg = repmat(neg_var, d-1, 1);

pos_train = -sum(((xtrain - repmat(pos_mean, size(train, 1), 1)) * V_pos).^2, 2)/2 - log(sum(S_pos))/2 + log(pi_pos);
neg_train = -sum(((xtrain - repmat(neg_mean, size(train, 1), 1)) * V_neg).^2, 2)/2 - log(sum(S_neg))/2 + log(pi_neg);

pos_test = -sum(((xtest - repmat(pos_mean, size(test, 1), 1)) * V_pos).^2, 2)/2 - log(sum(S_pos))/2 + log(pi_pos);
neg_test = -sum(((xtest - repmat(neg_mean, size(test, 1), 1)) * V_neg).^2, 2)/2 - log(sum(S_neg))/2 + log(pi_neg);

ytrain_label = max(sign(pos_train - neg_train), 0);
ytest_label = max(sign(pos_test - neg_test), 0);

err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end