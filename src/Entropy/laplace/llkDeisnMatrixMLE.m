function [loglikelihood,X,Y,mle] =llkDeisnMatrixMLE(data,mode,n,p,propSigma,ifprint)
%% Compute likelihood and construct design matrix and test data
loglikelihood = zeros(p,1);  % to store the logliklihood for each k
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
%            mle{k} = inv(X{k}'*X{k})*X{k}'*Y{k};
            mle{k} = GeneralMLE(X{k},Y{k},propSigma,'normal',0,5);
            loglikelihood(k) = LLK(X{k},Y{k},mle{k});
        end
    case 'MA'
end
if(ifprint)
    figure;
    plot(loglikelihood);
    title('log-likelihood');
end
end