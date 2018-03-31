function [bic] = BIC(llk,n,p,ifprint)
penalty = log(n-(1:p)).*(1:p);
bic = -2*llk'+penalty;
if(ifprint)
    figure;
    subplot(1,2,1)
    plot(penalty);
    title('BIC entropy penalty');
    subplot(1,2,2)
    plot(bic);
    title('BIC');
end