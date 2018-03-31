%% Model Selection for Limited data:
%% Numerical Experiments Using the New Criterion
%% Test on an Regression Case (Generalized Normal)

close all;
clc, clear;

%% Data generating parameters
n = 50;                            % number of observations 
p = 10;                            % max number of covariates (including intercept)
d = 4;                             % true dimension of the model
mu = 0;                            % GGN parameters, see https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function
b = 1/2;
rho = 4;                       
beta = zeros(p,1);                 % true parameter
% ind = sort(randperm(p,d));
% beta(ind) = randn(d,1);
beta(1:d) = randn(d,1);            % assume true dimension grows consecutively

% prior of the parameter beta is iid Normal(0,infty)
% bmean = 0;
% bsd = 1;


%% Generate data
X = randn(n,p-1);                  % design matrix
X = [ones(n,1),X];                 % add intercept into the design matrix
eps = gg6(1,n,mu,b,rho);           % iid normal noise
Y = X * beta + eps';

%% Calculate FIC (finite information criterion) with expanding parameter space
llk = zeros(1,p);
epy = zeros(1,p);
GGNpdf = @(y) b^(1/2)/2/2*gamma(1+1/rho)*exp(-b^(rho/2)*abs(y-mu)^rho);    % generalized normal pdf
for i = 1:p
   XX = X(:,1:i);                          % take out the design matrix of the dimension
   % calculate the MLE and the maximum likelihood
   fun = @(par) sum(abs(Y-XX*par).^rho);
   [beta_mle,fval,exitflag] = fminsearch(fun,zeros(i,1));
   while exitflag == 0
       [beta_mle,fval,exitflag] = fminsearch(fun,beta_mle);
   end
   llk(i) = -b^(rho/2)*fval;               % max log-likelihood (removing the common constants)
   % calculate the posterior entropy
   % using importance sampling
   N = 10000;
   sample = mvnrnd(zeros(1,i),eye(i),N);
   weight = zeros(N,1);
   logpiu = zeros(N,1);
   for j = 1:N
       logpiu(j) = -b^(rho/2)*fun(sample(j,:)');
       weight(j) = exp(logpiu(j)) / mvnpdf(sample(j,:),zeros(1,i),eye(i));
   end
   entropy = sum(weight.*logpiu)/sum(weight) - log(1/N*sum(weight));
   epy(i) = -2*entropy + i * log(2*pi*exp(1));
end

fic1 = -2*llk + epy;
bic = -2*(llk-1/2*log(n)*(1:p));
aic = -2*(llk-(1:p));

%% Plots
figure;
plot(llk);
title('log-likelihood');

figure;
subplot(2,2,1);
plot(epy);
title('entropy penalty');
subplot(2,2,2);
plot(log(n)*(1:10));
title('BIC penalty');
% subplot(2,2,3);
% plot(tva);
% title('trace of variance penalty');
subplot(2,2,4);
plot(2*(1:10));
title('AIC penalty');

figure;
subplot(2,2,1)
plot(fic1);
title('fic1');
subplot(2,2,2);
plot(bic);
title('bic');
% subplot(2,2,3);
% plot(fic2);
% title('fic2');
subplot(2,2,4);
plot(aic);
title('aic');
