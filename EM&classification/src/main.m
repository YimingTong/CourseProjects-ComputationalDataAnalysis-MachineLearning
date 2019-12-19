%This is the main routine for homework 4
%You are asked to plugin your implementation for the funciton ModelFull,
%ModelDiagonal, and ModelSpherical

%Repeat the experiments for 100 times
N = 100;

err_Full = zeros(N,2);
err_Diagonal = zeros(N,2);
err_Spherical = zeros(N,2);
err_KNN_5 = zeros(N,2);
err_KNN_10 = zeros(N,2);
err_KNN_15= zeros(N,2);
err_KNN_30 = zeros(N,2);
err_Logistic = zeros(N,2);

%Let p change from 0.1, 0.2, 0.5, 0.8, 0.9 to compare the performance of each classifier
p = 0.1;

for i = 1 : N
	
	[train, test] = SplitData(p);
	
	[err_train, err_test] = ModelFull(train, test);
	err_Full(i,:) = [err_train, err_test];
	
	[err_train, err_test] = ModelDiagonal(train, test);
	err_Diagonal(i,:) = [err_train, err_test];
	
	[err_train, err_test] = ModelSpherical(train, test);
	err_Spherical(i,:) = [err_train, err_test];
	
    [err_train, err_test] = KNN(train, test, 5);
	err_KNN_5(i,:) = [err_train, err_test];
    
    [err_train, err_test] = KNN(train, test, 10);
	err_KNN_10(i,:) = [err_train, err_test];

    [err_train, err_test] = KNN(train, test, 15);
	err_KNN_15(i,:) = [err_train, err_test];
    
    [err_train, err_test] = KNN(train, test, 30);
	err_KNN_30(i,:) = [err_train, err_test];
    
    [err_train, err_test] = Logistic(train, test);
	err_Logistic(i,:) = [err_train, err_test];
        	
end

mean_err_Full = mean(err_Full);
mean_err_Diagonal = mean(err_Diagonal);
mean_err_Spherical = mean(err_Spherical);
mean_err_KNN_5 = mean(err_KNN_5);
mean_err_KNN_10 = mean(err_KNN_10);
mean_err_KNN_15 = mean(err_KNN_15);
mean_err_KNN_30 = mean(err_KNN_30);
mean_err_Logistic = mean(err_Logistic);

fprintf('err_Full : %g, %g\n', mean_err_Full(1), mean_err_Full(2));
fprintf('err_Diagonal : %g, %g\n', mean_err_Diagonal(1), mean_err_Diagonal(2));
fprintf('err_Spherical : %g, %g\n', mean_err_Spherical(1), mean_err_Spherical(2));
fprintf('err_KNN_5 : %g, %g\n', mean_err_KNN_5(1), mean_err_KNN_5(2));
fprintf('err_KNN_10 : %g, %g\n', mean_err_KNN_10(1), mean_err_KNN_10(2));
fprintf('err_KNN_15 : %g, %g\n', mean_err_KNN_15(1), mean_err_KNN_15(2));
fprintf('err_KNN_30 : %g, %g\n', mean_err_KNN_30(1), mean_err_KNN_30(2));
fprintf('err_Logistic : %g, %g\n', mean_err_Logistic(1), mean_err_Logistic(2));
