function [ nts, loss1, ess ] = computeLoss( genTestX, genMufromX, genTrainX1, getW1 )
%return grid of training sizes and loss at each learner and training size

nMo = 3; %number of candidate models: linear (2param), quadratic, etc. 

%for numerically compute the loss 
nc = 1000; 
xc = genTestX(nc);
muc = genMufromX(xc);
Xc = cell(1,nMo); %design matrices
for m = 1:nMo 
    Xc{m} = [ones(nc,1)];
    for d=1:m
        Xc{m} = [Xc{m} (xc').^d];
    end
end

%under each learner and training size, gen obs. and obtain estimator, and then compute loss 
NT = 20;
loss1 = zeros(NT, nMo);
%nts = floor(50 * (1:NT)); %different sample sizes 
nts = 50:5:1000;
%ensure that different sample size corrsp. to the same realization 
x = genTrainX1(nts(end));
y =  genMufromX(x) + randn(1,nts(end)); %assume N(0,1) noises 
w = getW1(x);
W = diag(w);
phi = cell(1,nMo);
ess = zeros(1,length(nts));
count = 1;
for nt = nts %training size 
    x1 = x(1:nt);
    y1 = y(1:nt); %assume N(0,1) noises 
    W1 = W(1:nt,1:nt);
    ess(count) = getESS( w(1:nt) );
    %compute the loss at each nc and each learner 
    for m = 1:nMo 
        X = [ones(nt,1)];
        for d=1:m
            X = [X (x1').^d];
        end
        phi{m} = (X' * W1 * X) \ (X' * W1 * y1');
        muc_pred = Xc{m} * phi{m}; %predicted mean 
        loss1(count, m) = mean((muc' - muc_pred).^2);
    end
    count = count + 1;
end

end

