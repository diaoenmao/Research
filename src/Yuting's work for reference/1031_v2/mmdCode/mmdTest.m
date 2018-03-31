alpha = 0.05;
n = 18;
stat = zeros(3,n);
thresh = zeros(3,n);
stat_join =zeros(1,n);
thresh_join  = zeros(1,n);
params.sig = -1;
params.shuff = 3;
params.bootForce = 1;
for i=0:(n-1)
    tic
    f = csvread(sprintf('./csv/%d.csv',i));
    [tmpstat,tmpthresh,~] = mmdTestBoot(f(:,1),f(:,2),alpha,params);
    stat(1,i+1) = tmpstat;
    thresh(1,i+1) = tmpthresh;
    [tmpstat,tmpthresh,~] = mmdTestBoot(f(:,3),f(:,4),alpha,params);
    stat(2,i+1) = tmpstat;
    thresh(2,i+1) = tmpthresh;
    [tmpstat,tmpthresh,~] = mmdTestBoot(f(:,5),f(:,6),alpha,params);
    stat(3,i+1) = tmpstat;
    thresh(3,i+1) = tmpthresh;
    [tmpstat,tmpthresh,~] = mmdTestBoot([f(:,1) f(:,3 ) f(:,5)],[f(:,2) f(:,4) f(:,6)],alpha,params);
    stat_join(1,i+1) = tmpstat;
    thresh_join(1,i+1) = tmpthresh;
    toc
end
bresult =  stat>thresh;
bresult_join =  stat>thresh;
