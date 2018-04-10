import torch
from torch import nn
from torch.autograd import Variable
from util import *
from model import *
from modelselect import *
from data import *





max_num_epochs = 30    
dataSize = 1000
p=0.1
batch_size = 100
degree = 100
input_features = 10
ifcuda = True
input_datatype = torch.FloatTensor
target_datatype = torch.LongTensor
randomGen = np.random.RandomState(1)
X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
X_train, X_test, y_train, y_test = split_data_p(X,y,p=p,randomGen = randomGen)
X_train, X_test = X_train[:,:input_features], X_test[:,:input_features]
train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,batch_size)
model = LogisticRegression(input_features,2).cuda()
print_model(model)
free_parameters = get_free_parameters(model)
print(free_parameters)
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.LBFGS(free_parameters,lr=0.8) 
mw = modelWrapper(model)
optimizer_name = 'LBFGS'
optimizer_param = {'lr': 0.8}
mw.set_optimizer_name(optimizer_name)
mw.set_optimizer_param(optimizer_param)
mw.wrap()
TAG = 'TIC_test'



get_data_stats(X_train,TAG=TAG)
num_iter = 10
input = torch.from_numpy(X_train).type(input_datatype)
target = torch.from_numpy(y_train).type(target_datatype)
input,_ = normalize(input,TAG=TAG)
input = to_var(input,ifcuda)
target = to_var(target,ifcuda)

for i in range(num_iter):
    print('STEP: ', i)
    def closure():
        optimizer.zero_grad()
        # ===================forward=====================
        output = model(input)
        loss_batch = criterion(output, target)
        loss = torch.mean(loss_batch)
        # if(loss.data.cpu().numpy()<=1e-6):
            # for p in model.parameters():
                # print(p)
            # save_model(model,'./model/highorder.pth')
            # exit()    
            
        # L2 = get_TIC2(input,target,model)
        GTIC = get_GTIC(input.size()[0],model,loss_batch)
        L = loss + GTIC
        print(L)
        # for p in model.parameters():
            # print(p)
        # ===================backward====================
        L.backward()
        return L
    optimizer.step(closure)

  
#model = load_model(model,ifcuda,'./model/highorder.pth')
print_model(model)

train_output = model(input)
train_loss_batch = criterion(train_output, target)
print(torch.mean(train_loss_batch))
correct_cnt = get_correct_cnt(train_output,target,ifcuda) 
print(correct_cnt)
train_acc = correct_cnt/X_train.shape[0]
print(train_acc)

test_input = torch.from_numpy(X_test).type(input_datatype)
test_target = torch.from_numpy(y_test).type(target_datatype)
test_input,_ = normalize(test_input,TAG=TAG)
test_input = to_var(test_input,ifcuda)
test_target = to_var(test_target,ifcuda)

test_output = model(test_input)
test_loss_batch = criterion(test_output, test_target)
print(torch.mean(test_loss_batch))
correct_cnt = get_correct_cnt(test_output,test_target,ifcuda) 
print(correct_cnt)
test_acc = correct_cnt/X_test.shape[0]
print(test_acc)
                    
                    