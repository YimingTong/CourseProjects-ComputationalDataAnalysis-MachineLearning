house = readtable('RealEstate.csv');
ss = strcmp(house.Status, 'Short Sale');
fc = strcmp(house.Status, 'Foreclosure');
r = strcmp(house.Status, 'Regular');

%ridge regression
lambda = 0:0.001:100;
[nss,~] = size(house(ss,:));
[nfc,~] = size(house(fc,:));
[nr,~] = size(house(r,:));
err_ssr = zeros(nss, length(lambda));
err_fcr = zeros(nfc,length(lambda));
err_rr = zeros(nr, length(lambda));
K = 10;
dss = ceil(nss/K);
dfc = ceil(nfc/K);
dr = ceil(nr/K);
Xss = house{:,{'Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(ss,:);
yss = house.Price(ss);
Xfc = house{:,{'Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(fc,:);
yfc = house.Price(fc);
Xr = house{:,{'Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(r,:);
yr = house.Price(r);
for k = 1:K
    testIndss = ((k-1)*dss +1):min(dss*k,nss);
    trainIndss = setdiff(1:nss, testIndss);
    train_Xss = Xss(trainIndss,:);
    train_yss = yss(trainIndss);
    test_Xss = Xss(testIndss,:);
    test_yss = yss(testIndss,:);
    testIndfc = ((k-1)*dfc +1):min(dfc*k,nfc);
    trainIndfc = setdiff(1:nfc, testIndfc);
    train_Xfc = Xfc(trainIndfc,:);
    train_yfc = yfc(trainIndfc);
    test_Xfc = Xfc(testIndfc,:);
    test_yfc = yfc(testIndfc,:);
    testIndr = ((k-1)*dr +1):min(dr*k,nr);
    trainIndr = setdiff(1:nr, testIndr);
    train_Xr = Xr(trainIndr,:);
    train_yr = yr(trainIndr);
    test_Xr = Xr(testIndr,:);
    test_yr = yr(testIndr,:);
    b_ssr = ridge(train_yss, train_Xss, lambda,0);
    b_fcr = ridge(train_yfc, train_Xfc, lambda,0);
    b_rr = ridge(train_yr, train_Xr, lambda,0);
    y_pred_ssr = test_Xss*b_ssr(2:end,:) + repmat(b_ssr(1,:), [size(test_Xss, 1), 1]);
    y_pred_fcr = test_Xfc*b_fcr(2:end,:) + repmat(b_fcr(1,:), [size(test_Xfc, 1), 1]);
    y_pred_rr = test_Xr*b_rr(2:end,:) + repmat(b_rr(1,:), [size(test_Xr, 1), 1]);
    dist_ssr = bsxfun(@minus, y_pred_ssr, test_yss);
    dist_fcr = bsxfun(@minus, y_pred_fcr, test_yfc);
    dist_rr = bsxfun(@minus, y_pred_rr, test_yr);
    err_ssr(testIndss,:) = dist_ssr.^2;
    err_fcr(testIndfc,:) = dist_fcr.^2;
    err_rr(testIndr,:) = dist_rr.^2;
end

[~,i_ssr] = min(mean(err_ssr));
[~,i_fcr] = min(mean(err_fcr));
[~,i_rr] = min(mean(err_rr));
figure;plot(lambda, mean(err_ssr));
figure;plot(lambda, mean(err_fcr));
figure;plot(lambda, mean(err_rr));

%Lasso
[B_ss,FitInfo_ss] = lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(ss,:),house.Price(ss),'CV',10);
[B_fc,FitInfo_fc] = lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(fc,:),house.Price(fc),'CV',10);
[B_r,FitInfo_r] = lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(r,:),house.Price(r),'CV',10);

lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(ss,:),house.Price(ss),'Lambda',FitInfo_ss.LambdaMinMSE)
lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(fc,:),house.Price(fc),'Lambda',FitInfo_fc.LambdaMinMSE)
lasso(house{:,{'MLS','Bedrooms', 'Bathrooms','Size','Price_SQ_Ft'}}(r,:),house.Price(r),'Lambda',FitInfo_r.LambdaMinMSE)

lassoPlot(B_ss,FitInfo_ss,'PlotType','CV')
lassoPlot(B_fc,FitInfo_fc,'PlotType','CV')
lassoPlot(B_r,FitInfo_r,'PlotType','CV')

