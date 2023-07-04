close all; clear all; %clc;
bits = [2 4 8 16 32];
param.db_name ='MIRFLICKR';
param.km = [100 200 500]; % MIRFLICKR
param.mu = 3;

for i = 1:5
param.nbits = bits(i);
param.alpha = 1;
param.iter = 5; param.sf = 0.05;
param.omega = 10; param.beta = 0.01;
param.theta = 10; param.lambda = 0.001;

param.top_K = 2000;
param.pr_ind = [1:50:1000,1000];
param.pn_pos = [1:100:2000,2000];

fprintf('========Load data & Clustering======== \n');
[XTrain,YTrain,LTrain,XTest,YTest,LTest] = load_data(param.db_name);

LSet = cell(1,size(param.km,2));
for j = 1:size(param.km,2)
    [a,~] = kmeans(LTrain,param.km(j),'Distance','cosine');
    LSet{j} = sparse(1:size(LTrain,1),a,1);
    LSet{j} = full(LSet{j});
end

fprintf('========%s %d bits start======== \n', 'JSPSH',param.nbits);
evaluate_JSPSH(XTrain,YTrain,LTrain,XTest,YTest,LTest,LSet,param);
clearvars -except param bits
end
