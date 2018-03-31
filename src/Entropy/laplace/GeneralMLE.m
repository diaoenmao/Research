function mle = GeneralMLE(X,Y,propSigma,prior,varargin)
n = size(X,1);
p = size(X,2);
niter_mcmc = 1000;
nburnin = 250;
prior_m = 0;
prior_v = 1;
Nstart=prior_m;
totaliter = niter_mcmc+nburnin;
Nsample = zeros(p, totaliter);
Nsample(:,1)=Nstart;
for t = 2:totaliter
    Ncurrent = Nsample(:,t-1);
    for iD = 1:p
        beta_current = Ncurrent;
        Npropose = normrnd(beta_current(iD,:), propSigma)';
        beta_new = beta_current;
        beta_new(iD,:) = Npropose;
        prior_m = varargin{1};
        prior_v = varargin{2};
        post1 = LLK(X,Y,beta_new)-n*0.5*(1/prior_v*(beta_new-prior_m)'*(beta_new-prior_m));
        post2 = LLK(X,Y,beta_current)-n*0.5*(1/prior_v*(beta_current-prior_m)'*(beta_current-prior_m));
        pratio = post1-post2;     
        logU = log(rand());
        if(logU<pratio)
            Ncurrent = beta_new;
        end
    end
    Nsample(:,t) = Ncurrent;
end
Samples = Nsample(:,nburnin:end);
% hist(Samples);
mle = mode(Samples,2);
end