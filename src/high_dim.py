import torch
import config
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from modelWrapper import *

TAG = 'high_dim'
dataSize = 5000
input_features = 10
output_features = 2
config.init()
    
max_num_epochs = config.PARAM['max_num_epochs']
ifcuda = config.PARAM['ifcuda']
ifregularize = config.PARAM['ifregularize']
input_datatype = config.PARAM['input_datatype']
target_datatype = config.PARAM['target_datatype']
randomGen = np.random.RandomState(1)
X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
X_train, X_test, y_train, y_test = split_data_p(X,y,test_size=config.PARAM['test_size'],randomGen = randomGen)
X_train, X_test = X_train[:,:input_features], X_test[:,:input_features]
get_data_stats(X_train,TAG=TAG)
train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,config.PARAM['batch_size'])
test_loader = get_data_loader(X_test,y_test,input_datatype,target_datatype,config.PARAM['batch_size'])
list_train_loader = list(train_loader)

criterion = nn.CrossEntropyLoss(reduce=False)
model = local_Linear(input_features,output_features).type(input_datatype).cuda() if ifcuda else local_Linear(input_datatypes).type(config.PARAM['input_datatype'])
coordinate_set = model.coordinate_set(config.PARAM['local_size'])
mw = modelWrapper(model,config.PARAM['optimizer_name'])
mw.set_optimizer_param(config.PARAM['optimizer_param'])
mw.set_criterion(criterion)
mw.set_regularization(config.PARAM['regularization_parameters'],config.PARAM['if_joint_regularization'])
mw.wrap(coordinate_set)

train_loss_iter = []
train_regularized_loss_iter = []
train_acc_iter = []
        
optimizers = mw.optimizer
head_tracker = 0
stochastic_tracker = [None]*len(optimizers)
e = 0
while(e<=max_num_epochs):
    stochastic_tracker[1:] = stochastic_tracker[:-1]
    stochastic_tracker[0] = head_tracker    
    i = next((i for i, j in enumerate(stochastic_tracker) if j is not None), None)
    print(e)
    print(stochastic_tracker)
    print(i)
    if(i is None):
        break
    cur_tracker = stochastic_tracker[i]
    while(cur_tracker is not None and i<len(optimizers)):
        print(cur_tracker)
        optimizer = optimizers[i]
        input,target = list_train_loader[cur_tracker]
        input = to_var(input,ifcuda)
        target = to_var(target,ifcuda)        
        optimizer.zero_grad()
        loss,regularized_loss,_,acc = mw.loss_acc(input,target,ifregularize)
        train_loss_iter.append(float(loss))
        train_regularized_loss_iter.append(float(regularized_loss))
        train_acc_iter.append(float(acc))
        print('loss')
        print(loss)
        print('acc')
        print(acc)
        if(ifregularize):
            regularized_loss.backward()
        else:
            loss.backward()
        print('step')
        optimizer.step()
        i = i + 1
        if(i>=len(optimizers)):
            break
        cur_tracker = stochastic_tracker[i]
    if(head_tracker is None):
        head_tracker = None
    elif(head_tracker == len(list_train_loader)-1):
        head_tracker = 0
    else:
        head_tracker = head_tracker + 1
    if(stochastic_tracker[-1]==len(list_train_loader)-1):
        e = e + 1
    if(e==max_num_epochs):
        head_tracker =  None


final_train_loss = 0                    
final_train_acc = 0
total_train_size = 0
for input,target in train_loader:
    batch_size = input.size(0)
    total_train_size += batch_size
    input = to_var(input,ifcuda)
    target = to_var(target,ifcuda)   
    loss,_,_,acc = mw.loss_acc(input,target,ifregularize)
    final_train_loss += loss*batch_size
    final_train_acc += acc*batch_size
final_train_loss = float(final_train_loss/total_train_size)
final_train_acc = float(final_train_acc/total_train_size)
print('train loss: {}, train acc: {} of size {}'.format(final_train_loss,final_train_acc,total_train_size))

final_test_loss = 0                    
final_test_acc = 0
total_test_size = 0
for input,target in test_loader:
    batch_size = input.size(0)
    total_test_size += batch_size
    input = to_var(input,ifcuda)
    target = to_var(target,ifcuda)   
    loss,_,_,acc = mw.loss_acc(input,target,ifregularize)
    final_test_loss += loss*batch_size
    final_test_acc += acc*batch_size
final_test_loss = float(final_test_loss/total_test_size)
final_test_acc = float(final_test_acc/total_test_size)
print('test loss: {}, test acc: {} of size {}'.format(final_test_loss,final_test_acc,total_test_size))

