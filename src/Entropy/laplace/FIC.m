function [fic] = FIC(X,Y,llk,n,p,propSigma,mle,mode,ifprint,ifmcmcCheck,prior,varargin)
%% PARAM
niter_mcmc = 1000;
nburnin = 250;
prior_m = 0;
prior_v = 1;
Nstart=prior_m;
SamplesCell = cell(1,p);
CovCell = cell(1,p);
epy = zeros(p,1);  % to store the penalty term for each k 
if(ifprint&&ifmcmcCheck)
    mcmciter = p;
else
    mcmciter = 1:p;
end
%% MCMC(GIBBS) sampling to get the entropy
for k=mcmciter
    totaliter = niter_mcmc+nburnin;
    Nsample = zeros(k, totaliter);
    Nsample(:,1)=Nstart;
    for t = 2:totaliter 
        Ncurrent = Nsample(:,t-1);
        for iD = 1:k 
            beta_current = Ncurrent;
            Npropose = normrnd(beta_current(iD,:), propSigma)';
            beta_new = beta_current;
            beta_new(iD,:) = Npropose;
            switch prior
                case 'flat'
                    post1=-(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_new)'*(Y{k}-X{k}*beta_new);
                    post2 = -(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_current)'*(Y{k}-X{k}*beta_current);
                case 'mle'
                    prior_m = mle{k};
                    prior_v = (Y{k}-X{k}*prior_m)'*(Y{k}-X{k}*prior_m)/(n-k-k);
                    post1=-(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_new)'*(Y{k}-X{k}*beta_new)-(n-k)*0.5*(1/prior_v*(beta_new-prior_m)'*(beta_new-prior_m));
                    post2 = -(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_current)'*(Y{k}-X{k}*beta_current)-(n-k)*0.5*(1/prior_v*(beta_current-prior_m)'*(beta_current-prior_m));               
                case 'normal'
                    if(length(varargin)<2)
                        error('Specify mean and variance for normal prior')
                    end
                    prior_m = varargin{1};
                    prior_v = varargin{2};
                    post1=-(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_new)'*(Y{k}-X{k}*beta_new)-(n-k)*0.5*(1/prior_v*(beta_new-prior_m)'*(beta_new-prior_m));
                    post2 = -(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*beta_current)'*(Y{k}-X{k}*beta_current)-(n-k)*0.5*(1/prior_v*(beta_current-prior_m)'*(beta_current-prior_m));
                otherwise
                    error('Incompatible Prior')
            end
            pratio = post1-post2;     
            logU = log(rand());
            if(logU<pratio)
                Ncurrent = beta_new;
            end
        end
        Nsample(:,t) = Ncurrent;
    end
    if(ifprint&&ifmcmcCheck)
        figure
        plot(Nsample(1,nburnin:end))
        figure
        hold on;
        [acf,lags,bounds] = autocorr(Nsample(1,nburnin:end),100);
        stem(lags,acf); xlabel('Lag'); ylabel('\rho(k)');
        h = line(lags,bounds(1)*ones(length(acf),1));
        h1 = line(lags,bounds(2)*ones(length(acf),1));
        set(h,'color',[1 0 0]);
        set(h1,'color',[1 0 0]);
        hold off

    end
    SamplesCell{k} = Nsample(:,nburnin:end);
    CovCell{k} = cov(SamplesCell{k}');
    switch prior
        case 'flat'
            epy(k) = -log(det(CovCell{k}));
        case 'mle'
            epy(k) = k*log(prior_v)-log(det(CovCell{k}));
        case 'normal'
            epy(k) = k*log(prior_v)-log(det(CovCell{k}));
    end
end
fic = -2*llk + epy;
%% Print result
if(ifprint)
    figure;
    subplot(1,2,1)
    plot(epy);
    title('entropy penalty');
    subplot(1,2,2)
    plot(fic);
    title('fic');
end
end