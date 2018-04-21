import torch
from torch import nn
from torch.autograd import Variable
from util import *
from model import *
from ccd_model import *
from modelselect import *
from data import *





max_num_epochs = 10    
dataSize = 5000
test_size = 0.1
batch_size = 100
degree = 100
input_features = 10
output_features = 2
ifcuda = False
input_datatype = torch.FloatTensor
target_datatype = torch.LongTensor
randomGen = np.random.RandomState(1)
X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
X_train, X_test, y_train, y_test = split_data_p(X,y,test_size=test_size,randomGen = randomGen)
X_train, X_test = X_train[:,:input_features], X_test[:,:input_features]
train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,batch_size)
model = ccd_Linear(input_features,output_features).type(input_datatype)
#model = Linear(input_features,output_features,True).type(input_datatype)
param = list(model.parameters())
#print_model(model)
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.SGD(param[:15],lr=1e-1) 
TAG = 'TIC_test'



get_data_stats(X_train,TAG=TAG)

input = torch.from_numpy(X_train).type(input_datatype)
target = torch.from_numpy(y_train).type(target_datatype)
input,_ = normalize(input,TAG=TAG)
input = to_var(input,ifcuda)
target = to_var(target,ifcuda)

e = 0
i = 0
for epoch in range(max_num_epochs):
    for input, target in train_loader:
        input,_ = normalize(input,TAG=TAG)
        input = to_var(input,ifcuda)
        target = to_var(target,ifcuda)
        optimizer.zero_grad()
        output = model(input)
        loss_batch = criterion(output, target)
        loss = torch.mean(loss_batch)
        print(list(model.parameters())[0])
        #print(list(model.parameters())[21].grad)
        torch.autograd.grad(loss, param[:15],only_inputs=False,retain_graph=True)
        loss.backward()
        #print(list(model.parameters())[21].grad)
        print('loss')
        print(loss)
        optimizer.step()

#model = load_model(model,ifcuda,'./model/highorder.pth')
#print_model(model)

train_output = model(input)
train_loss_batch = criterion(train_output, target)
print(torch.mean(train_loss_batch))
train_acc = get_acc(train_output,target)
print(train_acc)

test_input = torch.from_numpy(X_test).type(input_datatype)
test_target = torch.from_numpy(y_test).type(target_datatype)
test_input,_ = normalize(test_input,TAG=TAG)
test_input = to_var(test_input,ifcuda)
test_target = to_var(test_target,ifcuda)

test_output = model(test_input)
test_loss_batch = criterion(test_output, test_target)
print(torch.mean(test_loss_batch))
test_acc = get_acc(test_output,test_target)
print(test_acc)
                    
                    