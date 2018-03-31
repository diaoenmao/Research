%Idea directions: 
%use my highdim regression + random covariate shift framework 
%design a phase transition situation to show a shape learning rate for
%different learners (consider Gaussian covariates)

%%
genMufromX = @(x) sin(10*x); %y | x
genTestX = @(n) rand(1,n); %covariate generating scheme for testing

%learner 1
genTrainX1 = @(n) -1+randn(1,n); %covariate generating scheme for training 
getW1 = @(x) exp((x+1).^2/2) .* (x>0 & x<1); %unnormalized weight, density ratio 
[nts loss1] = computeLoss( genTestX, genMufromX, genTrainX1, getW1 );

%learner 2
genTrainX2 = @(n) -0.5+randn(1,n);
getW2 = @(x) exp((x+0.5).^2/2) .* (x>0 & x<1); %unnormalized weight, density ratio 
[nts loss2] = computeLoss( genTestX, genMufromX, genTrainX2, getW2 );


%learner 3
u = 6;
genTrainX3 = @(n) -(u-1)+(u)*rand(1,n);
getW3 = @(x) (x>-(u-1) & x<1) .* (x>0 & x<1); %unnormalized weight, density ratio 
[nts, loss1, ess1] = computeLoss( genTestX, genMufromX, genTrainX3, getW3 );

% %learner 4
% u = 2;
% genTrainX4 = @(n) -(u-1)+(u)*rand(1,n);
% getW4 = @(x) (x>-(u-1) & x<1) .* (x>0 & x<1); %unnormalized weight, density ratio 
% [nts, loss2, ess2] = computeLoss( genTestX, genMufromX, genTrainX4, getW4 );


%% summary 
figure(1)
subplot(2,1,1)
plot(nts, loss1(:,1:3), 'o-')
title('learner 1')
xlabel('sample size')
ylabel('model')
legend('1', '2', '3')
ylim([0,1])

subplot(2,1,2)
plot(nts, ess1, 'o-')

figure(2)
subplot(2,1,1)
plot(nts, loss2(:,1:3), 'o-')
title('learner 2')
xlabel('sample size')
ylabel('model')
legend('1', '2', '3')
ylim([0,1])

subplot(2,1,2)
plot(nts, ess2, 'o-')
