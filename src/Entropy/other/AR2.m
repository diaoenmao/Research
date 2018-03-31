%%  

close all;
clc;
clear;
%% Data generating parameters
total = 2000;
n = 100;    % number of points
p = 10;     % range of test
d = 3;      % true generating model
prior_v = 0.01;

Evaluation_iterations = 100;
bestmodel_fic = zeros(1,Evaluation_iterations);
bestmodel_bic = zeros(1,Evaluation_iterations);
tic
for c=1:Evaluation_iterations
    %% Generate AR model with burning 

    X = zeros(n,1);
    X(1) = normrnd(0,1);
    X(2) = normrnd(0,1);
    X(3) = normrnd(0,1);

    for i = 4:total
        X(i) = 0.2*X(i-1)+0.1*X(i-2)+0.2*X(i-3)+normrnd(0,1);
    end

    Y = X(total-n+1:total);
    n = length(Y);    % number of points used in practice
    llk = zeros(p,1);  % to store the logliklihood for each k
    epy = zeros(p,1);  % to store the penalty term for each k 

    for k=1:p
        XX = [];    % Design matrix initial
        YY = Y(k+1:n);  
        for j=1:k

            XX = [Y(j:(n-1-k+j)),XX]; % Obtain the design matrix
        end
       b_mle = inv(XX'*XX)*XX'*YY;

       % Obtain the logliklihood by plug in 

            for j = (k+1):n
                llk(k) = llk(k)-0.5*log(2*pi)-0.5*(Y(j)-b_mle'*Y(j-1:-1:j-k))^2;
            end
        ss=0;
        for j=k+1:n
            ss = ss+Y(j-1:-1:j-k,:)*Y(j-1:-1:j-k,:)';
        end
        Sigma=inv(ss+1/prior_v*eye(k));
        epy(k) = log(det(prior_v*eye(k)))-log(det(Sigma));
%         Sigma=inv(ss);
%         epy(k) = -log(det(Sigma));
    end
    fic = -2*llk + epy;
    bic = -2*llk'+log(n).*(1:p);
    [~,bestmodel_fic(c)] = min(fic);
    [~,bestmodel_bic(c)] = min(bic);
% 
%     figure;
%     plot(llk);
%     title('log-likelihood');
% 
%     figure;
%     subplot(1,2,1)
%     plot(epy);
%     title('entropy penalty');
%     subplot(1,2,2)
%     plot(log(n).*(1:p));
%     title('BIC penalty');
% 
%     figure;
%     subplot(1,2,1)
%     plot(fic);
%     title('fic');
%     subplot(1,2,2)
%     plot(bic);
%     title('bic');
end
fic_prop=mean(bestmodel_fic == d);
bic_prop=mean(bestmodel_bic == d);
metric_m_fic = mean(bestmodel_fic);
metric_m_bic = mean(bestmodel_bic);
metric_std_fic = std(bestmodel_fic)/sqrt(Evaluation_iterations);
metric_std_bic = std(bestmodel_bic)/sqrt(Evaluation_iterations);
toc
   %% Importance sampling to get the entropy 
   
%    m = 1000;  % the total numbers of samples
%    w = zeros(m,1); 
%    Prior = normrnd(0,1,k,m);
%    
%    for j = 1:m
%        beta = Prior(:,j);
%        numerator = 1;
%     for i = (k+1):n
%             numerator=numerator*exp(Y(i)-beta'*Y(i-1:-1:i-k))^2;
%     end
%     w(j) = numerator;
%    end
%    
%    s = sum(w);
%    w = w./s;
% tmp = [0;cumsum(w)];
% betaresult = zeros(k,m);   
% for i=1:m
%        uniftest = rand();
%        idx = find(tmp<uniftest);
%        idx=idx(length(idx));
%        display(idx)
%        betaresult(:,i) = Prior(:,idx);
% end

