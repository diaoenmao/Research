
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
        X(i) = 0.2*X(i-1)+0.1*X(i-2)+0.2*X(i-3)+normrnd(0,1);
    end
    data = X(total-n+1:total);
    case 'MA'

    otherwise
        error('Incompatible Mode')
end
end