load('cs.mat');

% illustrate the picture
figure; imagesc(img); colormap gray;
A = randn(1300,2500);
y = A*double(img(:)) + 5*randn(1300,1);

% ridge regression
lambda = 50:10:200; % regularization parameter
n = length(lambda);
error = zeros(n, 1);
for i=1:n
    for j = 1:10
        yy=y;
        AA=A;
        yy(((j-1)*130+1):j*130) = [ ];
        AA(((j-1)*130+1):j*130, :) = [ ];
        co = ridge(yy,AA,lambda(i));
        e(j) = norm(co - double(img(:)))^2 / 2500;
    end
    error(i) = mean(e);
end
figure; plot(lambda, error);
[~, index] = min(error);
lambda_opt = lambda(index);
b_r = ridge(y,A,lambda_opt);
figure; imagesc(reshape(b_r,50,50));colormap gray; %recovered image by ridge regression

% lasso
lambda = 0:0.001:0.05;
n = length(lambda);
error = zeros(n, 1);
b_l = lasso(A, y,'Lambda',lambda, 'CV', 10);
for i=1:n
    error(i) = norm(b_l(:, i) - double(img(:)))^2 / 2500;
end
figure; plot(lambda, error);
[~, index] = min(error);
lambda_opt = lambda(index);
b_l = lasso(A,y,'Lambda',lambda_opt);
figure; imagesc(reshape(b_l,50,50));colormap gray; %recovered image by lasso