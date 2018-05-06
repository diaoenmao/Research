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


TAG = 'base_high_dim'
dataSize = 1100
input_features = 100
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
if_regularize = config.PARAM['if_regularize']
input_datatype = config.PARAM['input_datatype']
target_datatype = config.PARAM['target_datatype']
if_load = config.PARAM['if_load']
if_verbose = config.PARAM['if_verbose']
init_seed = 0
num_Experiments = 1
seeds = list(range(init_seed,init_seed+num_Experiments))

def main():
    if(if_load):
        remove_dir(['data/stats','output'])
    else:
        remove_dir(['data/stats','model','output'])
    for i in range(len(seeds)):
        runExperiment(seeds[i],'{}_{}'.format(TAG,i))
    process_output()
    return

def process_output():
    train_loss = np.zeros(num_Experiments)
    train_acc = np.zeros(num_Experiments)
    test_loss = np.zeros(num_Experiments)
    test_acc = np.zeros(num_Experiments)
    timing = np.zeros(num_Experiments)
    for i in range(num_Experiments):
        _,_,_,final_train_loss,final_train_acc,final_test_loss,final_test_acc,final_timing = load('./output/{}_{}.pkl'.format(TAG,i))
        train_loss[i] = final_train_loss
        train_acc[i] = final_train_acc
        test_loss[i] = final_test_loss
        test_acc[i] = final_test_acc
        timing[i] = final_timing
    print('Mean:\ntrain_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}, timing: {}'.format(np.mean(train_loss),np.mean(train_acc),np.mean(test_loss),np.mean(test_acc),np.mean(timing)))
    print('Stderr:\ntrain_loss: {}, train_acc:{}, test_loss: {}, test_acc: {}, timing: {}'.format(
        np.std(train_loss)/np.sqrt(num_Experiments),np.std(train_acc)/np.sqrt(num_Experiments),np.std(test_loss)/np.sqrt(num_Experiments),np.std(test_acc)/np.sqrt(num_Experiments),np.std(timing)/np.sqrt(num_Experiments)))
    return
    
def runExperiment(seed,TAG):
    s = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)

    high_dim = 10
    cov_mode = 'base'
    noise_sigma = np.sqrt(0.1)
    X, y = fetch_data_linear(dataSize,input_features,output_features,high_dim,cov_mode,noise_sigma,randomGen = randomGen)
    X_train, X_test, y_train, y_test = split_data_p(X,y,test_size=config.PARAM['test_size'],randomGen = randomGen)

    get_data_stats(X_train,TAG=TAG)
    if(config.PARAM['optimizer_name']=='LBFGS'):
        train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,device,X_train.shape[0])
    else:
        train_loader = get_data_loader(X_train,y_train,input_datatype,target_datatype,device,config.PARAM['batch_size'])
    test_loader = get_data_loader(X_test,y_test,input_datatype,target_datatype,device,config.PARAM['batch_size'])
    list_train_loader = list(train_loader)

    criterion = nn.CrossEntropyLoss(reduce=False) if if_classification else nn.MSELoss(reduce=False)
    model = Linear(input_features,output_features).to(device)
    if(if_load):
        model = load_model(model,'./model/{}.pth'.format(TAG))
    mw = modelWrapper(model,config.PARAM['optimizer_name'],device)
    mw.set_optimizer_param(config.PARAM['optimizer_param'],config.PARAM['reg_optimizer_param'])
    mw.set_criterion(criterion)   
    if(config.PARAM['regularization'] is not None):
        regularization = np.array(config.PARAM['regularization'])
        regularization_jitter = 0.1
        regularization = np.log(regularization + (randomGen.rand(regularization.shape[0])-0.5) * 2 * regularization_jitter+1e-6).tolist()
    else:
        regularization = config.PARAM['regularization']
    mw.set_regularization(regularization,config.PARAM['if_optimize_regularization'],config.PARAM['regularization_mode'])
    mw.wrap()
    
    eval(mw,train_loader,test_loader,TAG,if_classification)
    # param = list(mw.parameters())
    # print_param = [float(param[i]) for i in range(len(param))]
    # print(print_param)
    
    train_loss_iter = []
    train_regularized_loss_iter = []
    train_acc_iter = []
            
    optimizer = mw.optimizer
    for i in range(max_num_epochs):
        if(config.PARAM['optimizer_name']=='LBFGS'):
            for input,target in train_loader:
                input,_ = normalize(input,TAG=TAG) 
                def closure():
                    optimizer.zero_grad()
                    loss,regularized_loss = mw.L(input,target,False)
                    train_loss_iter.append(float(loss))
                    train_regularized_loss_iter.append(float(regularized_loss))
                    if(if_verbose):
                        print('loss')
                        print(float(loss))
                    if(if_classification):
                        acc = mw.acc(input,target)
                        train_acc_iter.append(float(acc))
                    if(if_regularize):
                        regularized_loss.backward()
                        return regularized_loss
                    else:
                        loss.backward()
                        return loss
                optimizer.step(closure)    
        else:
            for input,target in train_loader:
                input,_ = normalize(input,TAG=TAG)        
                optimizer.zero_grad()
                loss,regularized_loss = mw.L(input,target,False)
                train_loss_iter.append(float(loss))
                train_regularized_loss_iter.append(float(regularized_loss))
                if(if_verbose):
                    print('loss')
                    print(float(loss))
                if(if_classification):
                    acc = mw.acc(input,target)
                    train_acc_iter.append(float(acc))
                    if(if_verbose):
                        print('acc')
                        print(float(acc))
                if(if_regularize):
                    regularized_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

    e = time.time()
    final_timing = e-s
    print('Total Elapsed time: {}'.format(final_timing))
    param = list(mw.parameters())
    print_param = [float(param[i]) for i in range(len(param))]
    print(print_param)        
    if(config.PARAM['if_show']):
        show([train_loss_iter,train_regularized_loss_iter],['loss','regularized_loss'])
        if(if_classification):
            show([train_acc_iter],['acc'])

    final_train_loss,final_train_acc,final_test_loss,final_test_acc = eval(mw,train_loader,test_loader,TAG,if_classification)
    
    print('Experiment {} complete'.format(seed))
    if(config.PARAM['if_save']):
        save_model(mw.model, dir = './model/{}.pth'.format(TAG))
    save([train_loss_iter,train_regularized_loss_iter,train_acc_iter,final_train_loss,final_train_acc,final_test_loss,final_test_acc,final_timing],'./output/{}.pkl'.format(TAG))
    return final_train_loss,final_train_acc,final_test_loss,final_test_acc,final_timing
        

if __name__ == "__main__":
    main()