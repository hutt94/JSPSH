function [XTrain,YTrain,LTrain,XTest,YTest,LTest] = load_data(db_name)

load(['./datasets/',db_name,'.mat']);
rng(2023);

if strcmp(db_name, 'IAPRTC-12')
    clear V_tr V_te
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);

elseif strcmp(db_name, 'MIRFLICKR')
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);

elseif strcmp(db_name, 'NUSWIDE10')
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    clear I_tr I_te T_tr T_te L_tr L_te

elseif strcmp(db_name, 'MIRFLICKR_deep')
    X = (X-min(min(X)))/(max(max(X))-min(min(X)));
    R = randperm(size(X,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
end
clear X Y L PCA_Y R XAll


end