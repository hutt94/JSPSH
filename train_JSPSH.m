function [B,XW,YW] = train_JSPSH(XTrain,YTrain,LTrain,NLTrain,LSet,param)
    
    % parameters
    max_iter = param.iter;
    kdim = param.nbits/param.sf;
    omega = param.omega;
    theta = param.theta;
    beta = param.beta;
    alpha = param.alpha;
    lambda = param.lambda;
    mu = param.mu;
    nbits = param.nbits;
    sel_num = 1000;
    if strcmp(param.db_name, 'NUS-WIDE')
        sel_num = 5000;
    end
    
    n = size(LTrain,1);
    dX = size(XTrain,2);
    dY = size(YTrain,2);
    loss = zeros(1,max_iter);

    % hash code learning
    if kdim < n
        H = sqrt(n*nbits/kdim)*orth(rand(n,kdim));
        B = rsign(H,nbits,kdim);
        for i = 1:max_iter
            % update H
            BCluster = zeros(size(B));
            for j = 1: size(param.km,2)
                BCluster = BCluster + param.km(j)/sum(param.km)*LSet{j}*(LSet{j}'*B);
            end
            Z = omega * B + nbits * (NLTrain*(NLTrain'*B) + alpha*BCluster);
            [~,Lmd,VV] = svd(Z'*Z);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index));
            U = Z *  (V / (sqrt(Lmd(index,index))));
            U_ = orth(randn(n,kdim-length(find(index==1))));
            H = sqrt(n*nbits/kdim)*[U U_]*[V V_]';

            % update B
            BCluster = zeros(size(B));
            for j = 1: size(param.km,2)
                BCluster = BCluster + param.km(j)/sum(param.km)*LSet{j}*(LSet{j}'*H);
            end
            B = rsign(omega * H + nbits * (NLTrain*(NLTrain'*H) + alpha*BCluster)...
                    +theta*n*nbits/kdim*ones(n,kdim),nbits,kdim);
        end
    end
    clear Z Temp Lmd VV index U U_ V V_ BCluster

    % hash function learning
    sel_idx = randperm(size(LTrain,1),sel_num);
    Bs = B(sel_idx,:);
    YW = rand(dY,kdim);
    
    for i = 1:max_iter
        XW = (XTrain'*XTrain+(lambda)*eye(dX))\(XTrain'*B+ mu*XTrain'*YTrain*YW +((XTrain'*NLTrain)*NLTrain(sel_idx,:)')*Bs*beta*nbits)...
        /((1+mu) * eye(kdim)+Bs'*Bs*beta);
        YW = (YTrain'*YTrain+lambda*eye(dY))\(YTrain'*B+ mu*YTrain'*XTrain*XW +((YTrain'*NLTrain)*NLTrain(sel_idx,:)')*Bs*beta*nbits)...
        /((1+mu) * eye(kdim)+Bs'*Bs*beta);
    end

end