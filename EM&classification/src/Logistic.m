function [err_train, err_test] = Logistic(train, test)

[~, d] = size(train);

xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

xtrain = double(xtrain./255);
xtest = double(xtest ./255);

classifier = fitclinear(xtrain, ytrain, 'learner', 'logistic');

train_label = predict(classifier, xtrain);
test_label = predict(classifier, xtest);

err_train = sum(abs(train_label - (ytrain)))/length(ytrain);
err_test = sum(abs(test_label - ytest))/length(ytest);

end