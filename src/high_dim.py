import torch
import config
import time
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from modelWrapper import *

TAG = 'high_dim'
dataSize = 1000
input_features = 1000
output_features = 1
if(output_features==1):
    if_classification = False
elif(output_features>1):
    if_classification = True
else:
    print('invalid output features')
    exit()
config.init()

device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
ifregularize = config.PARAM['ifregularize']
input_datatype = config.PARAM['input_datatype']
target_datatype = config.PARAM['target_datatype']
if_GTIC = config.PARAM['if_GTIC']
randomGen = np.random.RandomState(1)

high_dim = 10
cov_mode = 'base'
noise_sigma = np.sqrt(0.1)
X, y = fetch_data_linear(dataSize,input_features,output_features,high_dim,cov_mode,noise_sigma,randomGen = randomGen)
X_train, X_test, y_train, y_test = split_data_p(X,y,test_size=config.PARAM['test_size'],randomGen = randomGen)

get_data_stats(X_train,TAG=TAG)
train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,device,config.PARAM['batch_size'])
test_loader = get_data_loader(X_test,y_test,input_datatype,target_datatype,device,config.PARAM['batch_size'])
list_train_loader = list(train_loader)

criterion = nn.CrossEntropyLoss(reduce=False) if if_classification else nn.MSELoss(reduce=False)
model = Linear(input_features,output_features).to(device)
coordinate_set = model.coordinate_set(config.PARAM['local_size'])
fixed_coordinate = model.fixed_coordinate()
mw = modelWrapper(model,config.PARAM['optimizer_name'],device)
mw.set_optimizer_param(config.PARAM['optimizer_param'])
mw.set_criterion(criterion)
mw.set_regularization(config.PARAM['regularization'],config.PARAM['if_optimize_regularization'],config.PARAM['regularization_mode'])
mw.set_coordinate(coordinate_set,fixed_coordinate)
mw.wrap()

train_loss_iter = []
train_regularized_loss_iter = []
train_acc_iter = []
        
optimizers = mw.optimizer
head_tracker = 0
stochastic_tracker = [None]*len(optimizers)
e = 0
param = list(mw.parameters())
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
        s2 = time.time()
        print(cur_tracker)
        coordinate = mw.coordinate_set[i]
        mw.activate_coordinate(coordinate)
        optimizer = optimizers[i]
        input,target = list_train_loader[cur_tracker]
        input,_ = normalize(input,TAG=TAG)        
        optimizer.zero_grad()
        s1 = time.time()
        loss,regularized_loss = mw.L(input,target,False,if_GTIC)
        train_loss_iter.append(float(loss))
        train_regularized_loss_iter.append(float(regularized_loss))
        print('loss')
        print(float(loss))
        if(if_classification):
            acc = mw.acc(input,target)
            train_acc_iter.append(float(acc))
            print('acc')
            print(float(acc))
        e1 = time.time()
        print(e1-s1)
        s3 = time.time()
        if(ifregularize):
            regularized_loss.backward()
        else:
            loss.backward()
        e3 = time.time()
        print(e3-s3)
        print('step')
        optimizer.step()
        i = i + 1
        if(i>=len(optimizers)):
            break
        cur_tracker = stochastic_tracker[i]
        e2 = time.time()
        print(e2-s2)
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
        
if(config.PARAM['ifshow']):
    show([train_loss_iter,train_regularized_loss_iter],['loss','regularized_loss'])
    show([train_acc_iter],['acc'])

final_train_loss = 0                    
final_train_acc = 0
total_train_size = 0
for input,target in train_loader:
    batch_size = input.size(0)
    input,_ = normalize(input,TAG=TAG)
    total_train_size += batch_size
    loss,_ = mw.L(input,target,True)
    final_train_loss += loss*batch_size
    if(if_classification):
        acc = mw.acc(input,target)
        final_train_acc += acc*batch_size
final_train_loss = float(final_train_loss/total_train_size)
final_train_acc = float(final_train_acc/total_train_size)
print('train loss: {}, train acc: {} of size {}'.format(final_train_loss,final_train_acc,total_train_size))

final_test_loss = 0                    
final_test_acc = 0
total_test_size = 0
for input,target in test_loader:
    batch_size = input.size(0)
    input,_ = normalize(input,TAG=TAG)
    total_test_size += batch_size  
    loss,_ = mw.L(input,target,True,if_classification=if_classification)
    final_test_loss += loss*batch_size
    if(if_classification):
        acc = mw.acc(input,target)
        final_test_acc += acc*batch_size
final_test_loss = float(final_test_loss/total_test_size)
final_test_acc = float(final_test_acc/total_test_size)
print('test loss: {}, test acc: {} of size {}'.format(final_test_loss,final_test_acc,total_test_size))

#save(mw, './model/model_{}.pth'.format(TAG))