clear;
close all;

% n=[20:10:100 150 200 400];
n=[20:20:100 150 200];
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


%Normal(0,1)
fic_prop_normal01 = zeros(1,nlength);
bic_prop_normal01 = zeros(1,nlength);
metric_m_fic_normal01 = zeros(1,nlength);
metric_m_bic_normal01 = zeros(1,nlength);
metric_std_fic_normal01 = zeros(1,nlength);
metric_std_bic_normal01 = zeros(1,nlength);
for i=1:nlength
    load(sprintf('iterresultn%dnormal01_notfull.mat',n(i)))
    fic_prop_normal01(i)=mean(iter_fic == d);
    bic_prop_normal01(i)=mean(iter_bic == d);
    metric_m_fic_normal01(i) = mean(iter_fic);
    metric_m_bic_normal01(i) = mean(iter_bic);
    metric_std_fic_normal01(i) = std(iter_fic)/sqrt(length(iter_fic));
    metric_std_bic_normal01(i) = std(iter_bic)/sqrt(length(iter_bic));
end

figure
subplot(2,2,1)
hold on
plot(n,bic_prop_flat)
plot(n,fic_prop_flat)
xlabel('sample size')
ylabel('Accuracy')
ylim([0,1])
title('Accuracy for flat prior')
legend('bic','fic')
subplot(2,2,2)
hold on
plot(n,bic_prop_normal01)
plot(n,fic_prop_normal01)
xlabel('sample size')
ylabel('Accuracy')
ylim([0,1])
title('Accuracy for normal prior')
legend('bic','fic')
subplot(2,2,3)
hold on
plot(n,metric_m_bic_flat)
plot(n,metric_m_fic_flat)
xlabel('sample size')
ylabel('mean')
ylim([1,10])
title('Accuracy for flat prior')
legend('bic','fic')
subplot(2,2,4)
hold on
plot(n,metric_m_bic_normal01)
plot(n,metric_m_fic_normal01)
xlabel('sample size')
ylabel('mean')
ylim([1,10])
title('Accuracy for normal prior')
legend('bic','fic')


