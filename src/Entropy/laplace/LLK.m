function loglikelihood = LLK(X,Y,beta)
n = size(X,1);
% p = size(X,2);
%normal
% loglikelihood = -n*0.5*log(2*pi)-0.5*(Y-X*beta)'*(Y-X*beta);
% Laplace
b=1;
loglikelihood = -n*log(2*b)-sum(abs(Y-X*beta)./b);
end