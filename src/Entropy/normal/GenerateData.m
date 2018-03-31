function data = GenerateData(mode,n)
total = 2000; 
%% Generate data with burning
switch mode
    case 'IID'
    case 'AR'
    X = zeros(n,1);
    X(1) = normrnd(0,1);
    X(2) = normrnd(0,1);
    X(3) = normrnd(0,1);
    for i = 4:total
% Normal noise
        noise = normrnd(0,1);
% Generate Laplacian noise
%         u = rand()-0.5;
%         mu = 0;
%         sigma = sqrt(2);
%         b = sigma / sqrt(2);
%         noise = mu - b * sign(u).* log(1- 2* abs(u));

        X(i) = 0.2*X(i-1)+0.1*X(i-2)+0.2*X(i-3)+noise;
    end
    data = X(total-n+1:total);
    case 'MA'

    otherwise
        error('Incompatible Mode')
end
end