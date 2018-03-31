 mu = [2,3];
sigma = [1,1.5;1.5,3];
rng default  % For reproducibility
r = mvnrnd(mu,sigma,100);
rng default  % For reproducibility
r2 = mvnrnd(mu,sigma,100);