%%  

close all;
clear;
%% Data generating parameters
total = 2000;
n = 100;    % number of points
p = 10;     % range of test
d = 3;      % true generating model
niter_mcmc = 1000;
nburnin = 250;
prior_m = 0;
prior_v = 1;
Nstart=prior_m;
propSigma = 0.1;
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
    SamplesCell = cell(1,p);
    CovCell = cell(1,p);
    for k=1:p
        XX = [];    % Design matrix initial
        YY = Y(k+1:n);  
        for j=1:k
           XX = [Y(j:(n-1-k+j)),XX]; % Obtain the design matrix
        end
       b_mle = inv(XX'*XX)*XX'*YY;

       % Obtain the logliklihood by plug in
%         for j = (p+1):n
%                 llk(k) = llk(k)-0.5*log(2*pi)-0.5*(Y(j)-b_mle'*Y(j-1:-1:j-k))^2;
%         end
       llk(k) = -(n-k)*0.5*log(2*pi)-0.5*(YY-XX*b_mle)'*(YY-XX*b_mle);
       %% MCMC sampling to get the entropy 
      totaliter = niter_mcmc+nburnin;

      Nsample = zeros(k, totaliter);
      Nsample(:,1)=Nstart;
    for t = 2:totaliter 
        Ncurrent = Nsample(:,t-1);
        for iD = 1:k 
            beta_current = Ncurrent';
            Npropose = normrnd(beta_current(:,iD), propSigma);
            beta_new = beta_current;
            beta_new(iD) = Npropose;
%             post1=0;
%             for j = (p+1):n
    %flat prior
%                 post1 = post1-0.5*log(2*pi)-0.5*(Y(j)-beta_new*Y(j-1:-1:j-k))^2;
    %normal prior
%                 post1 = post1-0.5*log(2*pi)-0.5*(Y(j)-beta_new*Y(j-1:-1:j-k))^2-0.5*(1/prior_v*(beta_new-prior_m)*(beta_new-prior_m)');
    %uniform prior
    %             tmp=zeros(1,length(beta_new));
    %             tmp(beta_new>=a&beta_new<=b) = beta_new(beta_new>=a&beta_new<=b);
    %             post1 = post1-0.5*log(2*pi)-0.5*(Y(j)-tmp*Y(j-1:-1:j-k))^2-0.5*(tmp*tmp');
%             end
    %flat prior
%             post1=-(n-k)*0.5*log(2*pi)-0.5*(YY-XX*beta_new')'*(YY-XX*beta_new');
    %normal prior
          post1=-(n-k)*0.5*log(2*pi)-0.5*(YY-XX*beta_new')'*(YY-XX*beta_new')-(n-k)*0.5*(1/prior_v*(beta_new-prior_m)*(beta_new-prior_m)');
            
%             post2=0;
%             for j = (p+1):n
    %flat prior
%                 post2 = post2-0.5*log(2*pi)-0.5*(Y(j)-beta_current*Y(j-1:-1:j-k))^2;
    %normal prior
%                 post2 = post2-0.5*log(2*pi)-0.5*(Y(j)-beta_current*Y(j-1:-1:j-k))^2-0.5*(1/prior_v*(beta_current-prior_m)*(beta_current-prior_m)');
    %uniform prior
    %              tmp=zeros(1,length(beta_current));
    %              tmp(beta_current>=a&beta_current<=b) = beta_current(beta_current>=a&beta_current<=b);
    %              post2 = post2-0.5*log(2*pi)-0.5*(Y(j)-tmp*Y(j-1:-1:j-k))^2-0.5*(tmp*tmp');
%             end
    %flat prior
%             post2 = -(n-k)*0.5*log(2*pi)-0.5*(YY-XX*beta_current')'*(YY-XX*beta_current');
    %normal prior
            post2 = -(n-k)*0.5*log(2*pi)-0.5*(YY-XX*beta_current')'*(YY-XX*beta_current')-(n-k)*0.5*(1/prior_v*(beta_current-prior_m)*(beta_current-prior_m)');
            pratio = post1-post2;     
            logU = log(rand());
            if(logU<pratio)
                Ncurrent = beta_new';
            end
        end
         Nsample(:,t) = Ncurrent;
    end
    
%     figure
%     plot(Nsample(1,nburnin:end))
%     figure
%     hold on;
%     [acf,lags,bounds] = autocorr(Nsample(1,nburnin:end),100);
%     stem(lags,acf); xlabel('Lag'); ylabel('\rho(k)');
%     h = line(lags,bounds(1)*ones(length(acf),1));
%     h1 = line(lags,bounds(2)*ones(length(acf),1));
%     set(h,'color',[1 0 0]);
%     set(h1,'color',[1 0 0]);
%     hold off

     SamplesCell{k} = Nsample(:,nburnin:end);
    % posmean=mean(Nsample,2);
    % poscov = Nsample-repmat(posmean,1,totaliter);
    % CovCell{k}=poscov*poscov'/niter_mcmc;
    CovCell{k} = cov(SamplesCell{k}');
    %flat prior
%     epy(k) = -log(det(CovCell{k}));
%     %normal prior
      epy(k) = k*log(prior_v)-log(det(CovCell{k}));
    end
    fic = -2*llk + epy;
    bic = -2*llk'+log(n-(1:p)).*(1:p);
    [~,bestmodel_fic(c)] = min(fic);
    [~,bestmodel_bic(c)] = min(bic);
end
fic_prop=mean(bestmodel_fic == d);
bic_prop=mean(bestmodel_bic == d);
metric_m_fic = mean(bestmodel_fic);
metric_m_bic = mean(bestmodel_bic);
metric_std_fic = std(bestmodel_fic)/sqrt(Evaluation_iterations);
metric_std_bic = std(bestmodel_bic)/sqrt(Evaluation_iterations);
toc

% load('iterresult.mat')
% iter_fic = [iter_fic bestmodel_fic];
% iter_bic = [iter_bic bestmodel_bic];
% save('iterresult.mat','iter_fic','iter_bic')

% figure;
% plot(llk);
% title('log-likelihood');
% 
% figure;
% subplot(1,2,1)
% plot(epy);
% title('entropy penalty');
% subplot(1,2,2)
% plot(log(n).*(1:p));
% title('BIC penalty');
% 
% figure;
% subplot(1,2,1)
% plot(fic);
% title('fic');
% subplot(1,2,2)
% plot(bic);
% title('bic');

% fic_prop=mean(iter_fic == d);
% bic_prop=mean(iter_bic == d);
% metric_m_fic = mean(iter_fic);
% metric_m_bic = mean(iter_bic);
% metric_std_fic = std(iter_fic)/sqrt(length(iter_fic));
% metric_std_bic = std(iter_bic)/sqrt(length(iter_bic));


% iter_fic =[];
% iter_bic =[];
% save('iterresult.mat','iter_fic','iter_bic')
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

