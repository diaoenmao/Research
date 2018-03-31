clear;
close all;
mode='AR';
% sample=[20:10:100 150 200 400 800];  % set of test sample size
%110
% sample=[120 130 140 160 170 180 190 250 300 350]; 
% sample=[40 60 100 150 200]; 
sample=[250 300 350]; 
p = 10;     % range of test dim
d = 3;      % true generating model
Evaluation_iterations = 200;
% ifprint = true;
ifprint = false;
% ifmcmcCheck = true;
ifmcmcCheck = false;
if(ifprint)
    sample = 100;
    Evaluation_iterations = 1;
end
propSigma = 0.3;
prior1 = 'flat';
prior2 = 'normal';
prior_m = 0;
prior_v = 1;
bestmodel_fic_flat = zeros(1,Evaluation_iterations);
bestmodel_fic_normal01 = zeros(1,Evaluation_iterations);
bestmodel_fic_normal51= zeros(1,Evaluation_iterations);
bestmodel_fic_normal04= zeros(1,Evaluation_iterations);
bestmodel_fic_normal00_1= zeros(1,Evaluation_iterations);
bestmodel_bic = zeros(1,Evaluation_iterations);
bestmodel_fic_normalmle = zeros(1,Evaluation_iterations);
bestmodel_fic_normal101 = zeros(1,Evaluation_iterations);
for i=1:length(sample)
    fprintf('Current sample size: %d\n',sample(i));
    tic
    for c=1:Evaluation_iterations
        fprintf('Current iteration: %d\n',c);
        data = GenerateData(mode,sample(i));
        n=length(data);
        [llk,X,Y,mle] =llkDeisnMatrixMLE(data,mode,n,p,propSigma,ifprint);
%         fic_flat=FIC(X,Y,llk,n,p,propSigma,mode,ifprint,ifmcmcCheck,prior1);
%         fic_normal01=FIC(X,Y,llk,n,p,propSigma,mode,ifprint,ifmcmcCheck,'mle');
%         fic_normal51=FIC(X,Y,llk,n,p,propSigma,mode,ifprint,ifmcmcCheck,prior2,10,1);
        fic_normalmle=FIC(X,Y,llk,n,p,propSigma,mle,mode,ifprint,ifmcmcCheck,'mle');
%         fic_normal101=FIC(X,Y,llk,n,p,propSigma,mle,mode,ifprint,ifmcmcCheck,prior2,10,1);
%         fic_normal04=FIC(X,Y,llk,n,p,propSigma,mode,ifprint,ifmcmcCheck,prior2,5,4);       
%         fic_normal00_1=FIC(X,Y,llk,n,p,propSigma,mode,ifprint,ifmcmcCheck,prior2,5,0.1); 
        bic=BIC(llk,n,p,ifprint);
%         [~,bestmodel_fic_flat(c)] = min(fic_flat);
        [~,bestmodel_fic_normalmle(c)] = min(fic_normalmle);
%         [~,bestmodel_fic_normal101(c)] = min(fic_normal101);
%         [~,bestmodel_fic_normal04(c)] = min(fic_normal04);
%         [~,bestmodel_fic_normal00_1(c)] = min(fic_normal00_1);
        [~,bestmodel_bic(c)] = min(bic);  
    end
    toc
     save2mat(bestmodel_bic,mode,'bic',sample(i))
%     save2mat(bestmodel_fic_flat,mode,'fic',sample(i),'flat')
    save2mat(bestmodel_fic_normalmle,mode,'fic',sample(i),'normal',1000,1000)
%     save2mat(bestmodel_fic_normal101,mode,'fic',sample(i),'normal',10,1)
%     save2mat(bestmodel_fic_normal04,mode,'fic',sample(i),'normal',0,4)
%     save2mat(bestmodel_fic_normal00_1,mode,'fic',sample(i),'normal',0,0.1)
end

  
