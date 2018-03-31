n=[30:20:90 400 800];
nlength = length(n);

% %Normal(10,1)
% for i=1:nlength
%     load(sprintf('./result2/ARficn%dnormal101.mat',n(i)))
%     iter_tmp = iter_ficnormal101;
%     load(sprintf('./result/ARficn%dnormal101.mat',n(i)))
%     iter_ficnormal101 = [iter_ficnormal101 iter_tmp];
%     save(sprintf('./result/ARficn%dnormal101.mat',n(i)),'iter_ficnormal101')
% end

%MLE
% for i=1:nlength
%     load(sprintf('./result2/ARficn%dnormal10001000.mat',n(i)))
%     iter_tmp = iter_ficnormal10001000;
%     load(sprintf('./result/ARficn%dnormal10001000.mat',n(i)))
%     iter_ficnormal10001000 = [iter_ficnormal10001000 iter_tmp];
%     save(sprintf('./result/ARficn%dnormal10001000.mat',n(i)),'iter_ficnormal10001000')
% end

%BIC
for i=1:nlength
    load(sprintf('./result2/ARbicn%d.mat',n(i)))
    iter_tmp = iter_bic;
    load(sprintf('./result/ARbicn%d.mat',n(i)))
    iter_bic = [iter_bic iter_tmp];
    save(sprintf('./result/ARbicn%d.mat',n(i)),'iter_bic')
end