%% Model Selection for Limited data:
%% Numerical Experiments Using the New Criterion
%% Test on an Linear Regression Case (Gaussian)

close all;
clc, clear;

%% Data generating parameters
n = 20;                            % number of observations 
p = 10;                            % max number of covariates (including intercept)
d = 4;                             % true dimension of the model
sig = 1;                           % sd of iid noise
beta = zeros(p,1);                 % true parameter
% ind = sort(randperm(p,d));
% beta(ind) = randn(d,1);
beta(1:d) = randn(d,1);            % assume true dimension grows consecutively

% prior of the parameter beta is iid Normal(0,1)
bmean = 0;
bsd = 1;


%% Generate data
X = randn(n,p-1);                  % design matrix
X = [ones(n,1),X];                 % add intercept into the design matrix
eps = sig * randn(n,1);            % iid normal noise
Y = X * beta + eps;

%% Calculate FIC (finite information criterion) with expanding parameter space
llk = zeros(1,p);
epy = zeros(1,p);
tva = zeros(1,p);
for i = 1:p
   XX = X(:,1:i);                          % take out the design matrix of the dimension
   b_mle = (XX'*XX) \ XX' * Y;             % MLE estimate
%   l = log(mvnpdf(Y',(XX*b_mle)',eye(n)));  % log-likelihood at MLE
   l = 0;
   for j = 1:n
       l = l - normlike([XX(j,:)*b_mle,1],Y(j));
   end
   llk(i) = l;
   Sigma = inv(XX' * XX + eye(i));
 %  Sigma = inv(XX' * XX);                   % prior: N(0,infty)
   tva(i) = 2*n*(trace(Sigma));
   h = 1/2 * log((2 * pi * exp(1))^i * det(Sigma));
   epy(i) = -2*h + i * log(2*pi*exp(1));
end

fic1 = -2*llk + epy;
fic2 = -2*llk + tva;
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
subplot(2,2,3);
plot(tva);
title('trace of variance penalty');
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
subplot(2,2,3);
plot(fic2);
title('fic2');
subplot(2,2,4);
plot(aic);
title('aic');
