clear;
close all;

n=[20 30 40 60 100 120 140 200:50:400];
% n=[20:10:100 150 200 400];
% n=[100];
d=3;
nlength = length(n);
%Flat
% fic_prop_flat = zeros(1,nlength);
% bic_prop_flat = zeros(1,nlength);
% metric_m_fic_flat = zeros(1,nlength);
% metric_m_bic_flat = zeros(1,nlength);
% metric_std_fic_flat = zeros(1,nlength);
% metric_std_bic_flat = zeros(1,nlength);
% for i=1:nlength
%     load(sprintf('./result/ARn%dflat_notfull.mat',n(i)))
%     fic_prop_flat(i)=mean(iter_fic == d);
%     bic_prop_flat(i)=mean(iter_bic == d);
%     metric_m_fic_flat(i) = mean(iter_fic);
%     metric_m_bic_flat(i) = mean(iter_bic);
%     metric_std_fic_flat(i) = std(iter_fic)/sqrt(length(iter_fic));
%     metric_std_bic_flat(i) = std(iter_bic)/sqrt(length(iter_bic));
% end


%Normal(10,1)
% fic_prop_normal101 = zeros(1,nlength);
% metric_m_fic_normal101 = zeros(1,nlength);
% metric_std_fic_normal101 = zeros(1,nlength);
% for i=1:nlength
%     load(sprintf('./result/ARficn%dnormal101.mat',n(i)))
%     fic_prop_normal101(i)=mean(iter_ficnormal101 == d);
%     metric_m_fic_normal101(i) = mean(iter_ficnormal101);
%     metric_std_fic_normal101(i) = std(iter_ficnormal101)/sqrt(length(iter_ficnormal101));
% end

%MLE
fic_prop_m_normal10001000 = zeros(1,nlength);
fic_prop_std_normal10001000 = zeros(1,nlength);
metric_m_fic_normal10001000 = zeros(1,nlength);
metric_std_fic_normal10001000 = zeros(1,nlength);
for i=1:nlength
    load(sprintf('./result/ARficn%dnormal10001000.mat',n(i)))
    fic_prop_m_normal10001000(i)=mean(iter_ficnormal10001000 == d);
    fic_prop_std_normal10001000(i)=std(iter_ficnormal10001000 == d)/sqrt(length(iter_ficnormal10001000));
    metric_m_fic_normal10001000(i) = mean(iter_ficnormal10001000);
    metric_std_fic_normal10001000(i) = std(iter_ficnormal10001000)/sqrt(length(iter_ficnormal10001000));
end

%BIC
bic_prop_m = zeros(1,nlength);
bic_prop_std = zeros(1,nlength);
metric_m_bic = zeros(1,nlength);
metric_std_bic = zeros(1,nlength);
for i=1:nlength
    load(sprintf('./result/ARbicn%d.mat',n(i)))
    bic_prop_m(i)=mean(iter_bic == d);
    bic_prop_std(i)=mean(iter_bic == d)/sqrt(length(iter_bic));
    metric_m_bic(i) = mean(iter_bic);
    metric_std_bic(i) = std(iter_bic)/sqrt(length(iter_bic));
end

n_interp = 20:400;
method = 'pchip';
bic_prop_m_interp=interp1(n,bic_prop_m,n_interp,method);
bic_prop_std_interp=interp1(n,bic_prop_std,n_interp,method);
metric_m_bic_interp=interp1(n,metric_m_bic,n_interp,method);
metric_std_bic_interp=interp1(n,metric_std_bic,n_interp,method);
fic_prop_m_normal10001000_interp=interp1(n,fic_prop_m_normal10001000,n_interp,method);
fic_prop_std_normal10001000_interp=interp1(n,fic_prop_std_normal10001000,n_interp,method);
metric_m_fic_normal10001000_interp=interp1(n,metric_m_fic_normal10001000,n_interp,method);
metric_std_fic_normal10001000_interp=interp1(n,metric_std_fic_normal10001000,n_interp,method);

fontsize = 16;
set(0,'defaultfigurecolor',[1 1 1])
figure
subplot(1,2,1)
hold on
grid on
% BIC=shadedErrorBar(n,bic_prop_m,bic_prop_std,{'-r','LineWidth',1.5});
% LoL=shadedErrorBar(n,fic_prop_m_normal10001000,fic_prop_std_normal10001000,{'-b','LineWidth',1.5});
BIC=shadedErrorBar(n_interp,bic_prop_m_interp,bic_prop_std_interp,{'-r','LineWidth',1.5});
 LoL=shadedErrorBar(n_interp,fic_prop_m_normal10001000_interp,fic_prop_std_normal10001000_interp,{'-b','LineWidth',1.5});
set(gca,'Fontsize',fontsize)
L1=legend([BIC.mainLine,LoL.mainLine],'BIC','LoL');
set(L1,'FontSize',fontsize); 
xlabel('sample size','FontSize', fontsize)
ylabel('Accuracy','FontSize', fontsize)
ylim([0,1])
title('Accuracy','FontSize', fontsize)


subplot(1,2,2)
hold on
grid on
% BIC=shadedErrorBar(n,metric_m_bic,metric_std_bic,{'-r','LineWidth',1.5});
% LoL=shadedErrorBar(n,metric_m_fic_normal10001000,metric_std_fic_normal10001000,{'-b','LineWidth',1.5});
BIC=shadedErrorBar(n_interp,metric_m_bic_interp,metric_std_bic_interp,{'-r','LineWidth',1.5});
LoL=shadedErrorBar(n_interp,metric_m_fic_normal10001000_interp,metric_std_fic_normal10001000_interp,{'-b','LineWidth',1.5});
set(gca,'Fontsize',fontsize)
L2=legend([BIC.mainLine,LoL.mainLine],'BIC','LoL');
set(L2,'FontSize',fontsize); 
xlabel('sample size','FontSize', fontsize)
ylabel('Mean','FontSize', fontsize)
ylim([1,10])
title('Mean','FontSize', fontsize)





