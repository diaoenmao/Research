import torch
import numpy as np
from util import *
from data import *
from model import *

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

dataSize = 200
local_size = 5
input_features = 10
output_features = 2
max_num_epochs = 5
randomGen = np.random.RandomState(1)
input_datatype = torch.FloatTensor
X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
dataX = X[:,:input_features]
dataY = y
model = ccd_Linear(input_features,output_features).type(input_datatype).cuda()
param = list(model.parameters())
print(param)
# splitted_param_0 = list(torch.split(param[0],local_size,0))
# print(splitted_param_0[0]._grad_fn)
# splitted_param_0[0].detach_()
# print(splitted_param_0[0]._grad_fn)
# print(param[0].requires_grad)
optimizer = torch.optim.LBFGS(param[:5], lr = 0.3)
# for epoch in range(max_num_epochs):
    # def closure():
        # optimizer.zero_grad()
        # # ===================forward=====================
        # loss,regularized_loss,loss_batch,acc = modelwrapper.loss_acc(input,target,self.ifregularize)
        # output = model(input)
        # loss_batch = self.criterion(output, target)
        # loss = torch.mean(loss_batch)
        # # ===================backward====================
        # if(self.ifregularize):
            # print('a')
            # print(modelwrapper.parameters())
            # print(loss)
            # print(regularized_loss)
            # regularized_loss.backward()
            # return regularized_loss
        # else:
            # loss.backward()
            # return loss
    # optimizer.step(closure)       
    # e = time.time()
            
exit()






