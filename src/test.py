import torch
import numpy as np
import time
from util import *
from data import *
from torch.distributions.multivariate_normal import MultivariateNormal 
from torch.distributions.normal import Normal
from sklearn.datasets import load_boston
from torch import nn
import torch.nn.functional as F


# X, y = fetch_data_mld()
# valid_target = np.array([5])
# X, y = filter_data(X,y,valid_target)
# first_image = X[0,:]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.ion()
# plt.figure()
# plt.imshow(pixels, cmap='gray')
# plt.show()



# first_image = X[0,:]  
# first_image = np.array(first_image, dtype='float')      
# dims = (28,28)
# input_features = gen_input_features_LR(dims)
# print(len(input_features))
# for i in range(len(input_features)):
    # cur_img = first_image[input_features[i]].reshape((np.int(np.sqrt(input_features[i].shape[0])),np.int(np.sqrt(input_features[i].shape[0]))))
    # plt.figure()
    # plt.imshow(cur_img, cmap='gray')
    # plt.show()
    
# plt.pause(10000)


# paths=['model','output']
# zip_name = './tmp/Experiment_{}.zip'.format(100)
# zip_dir(paths,zip_name)
#remove_dir(paths)

#'Base_1','AIC_1','BIC_1','BC_1','CrossValidation_1','CrossValidation_3','CrossValidation_10','GTIC_1','Lasso_1','Ridge_1','ElasticNet_1','GREG_1'
# mode = 'BIC_1'
# selected_model_id,best_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency,timing = load('./output/Experiment_{}.pkl'.format(mode))
# print(efficiency)
# print(np.mean(efficiency[efficiency!=0]))
# print(np.std(efficiency[efficiency!=0]))

# dataSize = 200
# local_size = 5
# input_features = 10
# output_features = 2
# max_num_epochs = 5
# randomGen = np.random.RandomState(1)
# input_datatype = torch.FloatTensor
# X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
# dataX = X[:,:input_features]
# dataY = y
# model = ccd_Linear(input_features,output_features).type(input_datatype).cuda()
# param = list(model.parameters())
# print(param)
# # splitted_param_0 = list(torch.split(param[0],local_size,0))
# # print(splitted_param_0[0]._grad_fn)
# # splitted_param_0[0].detach_()
# # print(splitted_param_0[0]._grad_fn)
# # print(param[0].requires_grad)
# optimizer = torch.optim.LBFGS(param[:5], lr = 0.3)
# # for epoch in range(max_num_epochs):
    # # def closure():
        # # optimizer.zero_grad()
        # # # ===================forward=====================
        # # loss,regularized_loss,loss_batch,acc = modelwrapper.loss_acc(input,target,self.ifregularize)
        # # output = model(input)
        # # loss_batch = self.criterion(output, target)
        # # loss = torch.mean(loss_batch)
        # # # ===================backward====================
        # # if(self.ifregularize):
            # # print('a')
            # # print(modelwrapper.parameters())
            # # print(loss)
            # # print(regularized_loss)
            # # regularized_loss.backward()
            # # return regularized_loss
        # # else:
            # # loss.backward()
            # # return loss
    # # optimizer.step(closure)       
    # # e = time.time()
            
# exit()

# TAG ='high_dim'
# num_Experiment = 100
# l_final_train_loss = np.zeros(num_Experiment)
# l_final_test_loss = np.zeros(num_Experiment)
# for i in range(num_Experiment):
    # train_loss_iter,train_regularized_loss_iter,train_acc_iter,final_train_loss,final_train_acc,final_test_loss,final_test_acc,final_timing = load('./output/{}_{}.pkl'.format(TAG,i))
    # l_final_train_loss[i] = final_train_loss
    # l_final_test_loss[i] = final_test_loss
# print(np.mean(l_final_train_loss[l_final_train_loss<2]))
# print(np.mean(l_final_test_loss[l_final_test_loss<2]))

# a = np.random.randn(100,5)
# b = np.matmul(a.T,a)
# print(b)
# a_1 = np.expand_dims(a, axis=2)
# a_2 = np.transpose(a_1,(0,2,1))
# print(a_1.shape)
# print(a_2.shape)
# a_3 = np.sum(np.matmul(a_1,a_2),0)
# print(a_3)



# t = np.arange(0,10,1/50)
# signal1 = np.sin(2*np.pi*15*t) + np.sin(2*np.pi*20*t)
# signal2 = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*5*t)
# signal = np.vstack((signal1,signal2))
# # plt.figure()
# # plt.plot(t,signal)
# # plt.show()
# sp = np.fft.fft(signal,axis=1)
# mag_sp = np.absolute(sp)
# f = np.arange(0,sp.shape[1])*50/sp.shape[1]
# plt.figure()
# plt.plot(f,mag_sp[1,:])
# plt.show()

# def is_pos_def(x):
    # return np.all(np.linalg.eigvals(x) > 0)

# def entropy(var):
    # if(len(var.shape)>1):
        # sign, logdet = np.linalg.slogdet(2*np.pi*np.e*var)
        # entropy = 0.5*sign*logdet
    # else:
        # entropy = 0.5*(np.log(2*np.pi*np.e*var))
    # return entropy
    
# t = 1    
# batch_size = 1000
# dim = 5
# p=np.ones(dim)*1
# p[1] = 0.2 
# x = np.random.randn(batch_size,dim)
# x[:,1] = 0.1*np.power(x[:,0],1)+0.3*np.power(x[:,2],1)+0.001*np.random.rand(batch_size)
# z = np.random.rand(batch_size,dim)<=p
# cov_x = np.cov(x.T)
# x = x * z / p
# mean_x = np.mean(x,axis=0)
# cov_x = np.cov(x.T)

# mean_x = torch.from_numpy(np.mean(x,axis=0))
# cov_x = torch.from_numpy(np.cov(x.T))
# print(mean_x)
# print(cov_x)

# marg_cov_x = torch.diag(torch.diagonal(cov_x))
# mvn = MultivariateNormal(mean_x,covariance_matrix=cov_x)
# marg_mvn = MultivariateNormal(mean_x,covariance_matrix=marg_cov_x)
# joint_entropy = mvn.entropy()
# marg_entropy = marg_mvn.entropy()
# total_correlation = marg_entropy - joint_entropy
# print(joint_entropy)
# print(marg_entropy)
# print(total_correlation)

# s = time.time()
# metric = np.zeros(dim)
# for r in range(t):
    # for i in range(dim):
        # entropy_this = entropy(cov_x[i,i])
        # tmp = np.delete(cov_x,i,0)
        # cov_other = np.delete(tmp,i,1)
        # entropy_marg = entropy(np.diagonal(cov_x))
        # entropy_other_marg = np.delete(entropy_marg,i,0)
        # entropy_other = entropy(cov_other)
        # entropy_joint = entropy(cov_x)

        # # print(entropy_this)
        # # print(entropy_marg)
        # # print(entropy_other)
        # print(entropy_joint)

        # metric[i] = entropy_this-(entropy_joint-entropy_other)
        
    # print(metric)
# e = time.time()
# print("Elapsed Time:", (e-s))

# class LinearRegression(nn.Module):

    # def __init__(self, input_dim, output_dim):
        # super(LinearRegression, self).__init__() 
        # self.linear = nn.Linear(input_dim, output_dim)

    # def forward(self, x):
        # out = self.linear(x)
        # return out

# class GatedLinearRegression(nn.Module):

    # def __init__(self, input_dim, output_dim):
        # super(GatedLinearRegression, self).__init__() 
        # self.linear = nn.Linear(input_dim, output_dim)
        # self.feature_select = nn.Linear(input_dim, output_dim)
    # def forward(self, x):
        # fs = self.feature_select(x)
        # map = F.sigmoid(fs)
        # x = self.linear(x)
        # x = x*map/torch.mean(map)
        # return x
        
# class DropoutLinearRegression(nn.Module):

    # def __init__(self, input_dim, output_dim):
        # super(DropoutLinearRegression, self).__init__() 
        # self.linear = nn.Linear(input_dim, output_dim)
        # self.dp = nn.Dropout(0.01)
    # def forward(self, x):
        # x = self.linear(x)
        # x = self.dp(x)
        # return x
 
# class NeuralNet(nn.Module):
    # def __init__(self, input_size, hidden_size, num_classes):
        # super(NeuralNet, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size) 
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    # def forward(self, x):
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # return out
        
# model = LinearRegression(13,1) 
# #model = GatedLinearRegression(13,1)        
# #model = DropoutLinearRegression(13,1)
# #model = NeuralNet(13,5,1)
# criterion = nn.MSELoss()
# lr = 1e-4
# optimiser = torch.optim.SGD(model.parameters(), lr = lr)

# epochs = 100
# seed = 1
# input,target = load_boston(return_X_y=True)
# X_train, X_test, y_train, y_test = split_data_p(input,target,randomGen = np.random.RandomState(seed))
# torch.from_numpy(X_train).to(torch.float32)
# target = torch.from_numpy(y_train).to(torch.float32)

# for epoch in range(epochs):

    # optimiser.zero_grad()
    # output = model.forward(input).squeeze()
    # loss = criterion(output, target)
    # loss.backward()
    # optimiser.step()
    # with torch.no_grad():
        # input = torch.from_numpy(X_test).to(torch.float32)
        # target = torch.from_numpy(y_test).to(torch.float32)
        # output = model.forward(input).squeeze()
        # loss = criterion(output, target)
        # print('epoch: {}, loss: {}'.format(epoch,loss.item()))





