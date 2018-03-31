function [llk,X,Y,mle] =likelihood(data,mode,n,p,ifprint)
%% Compute likelihood and construct design matrix and test data
llk = zeros(p,1);  % to store the logliklihood for each k
switch mode
    case 'IID'
    case 'AR'
        X = cell(1,p);    % Design matrix initial
        Y = cell(1,p);    % test data initial
        mle = cell(1,p);
        for k=1:p
            Y{k} = data(k+1:n);
            for j=1:k
               X{k} = [data(j:(n-1-k+j)),X{k}]; % Obtain the design matrix
            end
           mle{k} = inv(X{k}'*X{k})*X{k}'*Y{k};
           llk(k) = -(n-k)*0.5*log(2*pi)-0.5*(Y{k}-X{k}*mle{k})'*(Y{k}-X{k}*mle{k});
        end
    case 'MA'
end
if(ifprint)
    figure;
    plot(llk);
    title('log-likelihood');
end
end