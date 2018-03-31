import numpy
import theano
import theano.tensor as T
import limlearn as lol
import limlearn.cv,limlearn.mlp

import sys
import time
import os.path
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from data import *
from train import *
from sl import *

def runExperiment(dataSizes,num_hidden_nodes_list):
    oracleSize = 10000
    ifload = True
    models_dir = './model/models.pkl'
    TIC_dir = './model/tic.pkl'
    if(os.path.exists(models_dir) and ifload):
        models = load(models_dir) 
    else:
        models=[]
        for i in range(len(num_hidden_nodes_list)):
            models.append(build_sk_mlp(num_hidden_nodes_list[i]))
        save(models,models_dir)
        
    models_cv=[]   
    for i in range(len(num_hidden_nodes_list)):
            models_cv.append(build_sk_mlp(num_hidden_nodes_list[i],False))
            
    max_num_hidden_layer = 1
    num_hidden_layer_list = []
    for i in range(len(num_hidden_nodes_list)):
        num_hidden_layer_list.append(len(num_hidden_nodes_list[i]))
    max_num_hidden_layer = max(num_hidden_layer_list)
    num_hidden_layers = list(range(1,max_num_hidden_layer+1)) 
    
    if(os.path.exists(TIC_dir) and ifload):
        tensorslist,num_hidden_layer_list = load(TIC_dir) 
    else:
        tensorslist = limlearn.mlp.BuildModelClass(num_hidden_layers)
        save([tensorslist,num_hidden_layer_list],TIC_dir)
    
    tics = []
    for i in range(len(tensorslist)):   
        tensors = tensorslist[i]
        input = [tensors[-2],tensors[-1]]
        tic = lol.tic(*input)     
        tics.append(theano.function(inputs=tensors[:-2],outputs=tic))
                  
    oracleX,oracleY = generate_data(oracleSize)
    required_tics = [tics[i-1] for i in num_hidden_layer_list]
    loss_oracle = np.zeros((len(models),len(dataSizes)))
    timing = {'tic': [], 'holdout': [], '3fold': [], '10fold': [], 'loo': []}
    model_selected = {'tic': [], 'holdout': [], '3fold': [], '10fold': [], 'loo': []}
    for i in range(len(dataSizes)):
        print(dataSizes[i])
        dataX,dataY = generate_data(dataSizes[i])
        
        thresh = np.floor((np.sqrt(dataSizes[i])/(dataX.shape[1]+1))-1).astype(np.int)
        active = np.squeeze(num_hidden_nodes_list<=thresh)
        active_models_cv = list(compress(models_cv, active))

        for j in range(len(models)):
            models[j] = train_sk_mlp(dataX,dataY,models[j])
   
        start=time.time()
        active_models = list(compress(models, active))
        tic_tmp = run_tic(dataX,dataY,active_models,required_tics)
        model_selected['tic'].append(tic_tmp)
        end=time.time()
        timing['tic'].append(tmp_time+end-start)
        
        start=time.time()
        model_selected['holdout'].append(run_holdout(dataX,dataY,active_models_cv))
        end=time.time()
        timing['holdout'].append(end-start)
        
        start=time.time()
        model_selected['3fold'].append(run_kfold(dataX,dataY,3,active_models_cv))
        end=time.time()
        timing['3fold'].append(end-start)
        
        start=time.time()
        model_selected['10fold'].append(run_kfold(dataX,dataY,10,active_models_cv))
        end=time.time()
        timing['10fold'].append(end-start)
                
        start=time.time()
        model_selected['loo'].append(run_loo(dataX,dataY,active_models_cv))
        end=time.time()
        timing['loo'].append(end-start)

        for j in range(len(models)):
            loss_oracle[j,i] = loss_sk_mlp(oracleX,oracleY,models[j])
            
        print([timing['tic'],timing['holdout'],timing['3fold'],timing['10fold'],timing['loo']])
    return loss_oracle,model_selected,timing
    
def main():
    result_dir = 'result.pkl'
    init_dataSize = 200
    batch_dataSize = 10
    end_dataSize = 1000
    dataSizes = range(init_dataSize,end_dataSize+1,batch_dataSize)
    max_num_hidden_layer = 2
    min_num_hidden_nodes = 1
    max_num_hidden_nodes = 10
    first_layer = [[i] for i in range(1,max_num_hidden_nodes+1)]
    num_hidden_nodes_list = [[i] for i in range(1,max_num_hidden_nodes+1)]
    # for i in range(1,max_num_hidden_nodes+1):
        # for j in range(1,6):            
            # num_hidden_nodes_list.append([i,j])
    loss_oracle,model_selected,timing = runExperiment(dataSizes, num_hidden_nodes_list)
    opt_loss_oracle = np.min(loss_oracle,axis=0)
    loss_ratio = {'tic':  None, 'holdout': None, '3fold': None, '10fold': None, 'loo': None}
    mode = ['tic','holdout','3fold','10fold','loo']
    for i in range(len(mode)):
        #print(model_selected[mode[i]])
        #print(loss_oracle[model_selected[mode[i]],np.arange(loss_oracle.shape[1])])
        loss_ratio[mode[i]] = loss_oracle[model_selected[mode[i]],np.arange(loss_oracle.shape[1])]/opt_loss_oracle
        print(np.mean(loss_ratio[mode[i]]))
    print(loss_ratio)
    #viewResult(dataSizes,loss_ratio,mode)
    #viewResult(dataSizes,timing,mode)
    save([dataSizes,timing,mode],result_dir) 
    
# def set_hidden_nodes_list(max_num_hidden_layer,min_num_hidden_nodes,max_num_hidden_nodes):
    # for i in range(1,max_num_hidden_layer+1):
    
def run_tic(dataX,dataY,models,tics):
    lossAndTIC = np.zeros(len(models))
    for i in range(len(models)):
        input = [dataX, dataY]
        param = param_sk_mlp(models[i])
        input.extend(param)  
        lossAndTIC[i] = loss_sk_mlp(dataX,dataY,models[i]) + tics[i](*input)
    return np.argmin(lossAndTIC)
    
def run_holdout(dataX,dataY,models):
    dataX_train, dataX_validation, dataY_train, dataY_validation=lol.cv.holdout(dataX,dataY, 0.7)
    loss = np.zeros(len(models))
    for i in range(len(models)):
        models[i].fit(dataX_train, dataY_train)
        loss[i] = loss_sk_mlp(dataX_validation,dataY_validation,models[i])
    return np.argmin(loss)

def run_kfold(dataX,dataY,k,models):
    dataX_train, dataX_validation, dataY_train, dataY_validation=lol.cv.kfold(dataX,dataY,k)
    loss = np.zeros((len(models),len(dataX_train)))
    for i in range(len(models)):
        for j in range(len(dataX_train)):
            models[i].fit(dataX_train[j], dataY_train[j])
            loss[i,j]=loss_sk_mlp(dataX_validation[j], dataY_validation[j],models[i])
    return np.argmin(np.min(loss,axis=1))

def run_loo(dataX,dataY,models):
    dataX_train, dataX_validation, dataY_train, dataY_validation=lol.cv.loo(dataX,dataY)
    loss = np.zeros((len(models),len(dataX_train)))
    for i in range(len(models)):
        for j in range(len(dataX_train)):
            models[i].fit(dataX_train[j], dataY_train[j])
            loss[i,j]=loss_sk_mlp(dataX_validation[j], dataY_validation[j],models[i])
    return np.argmin(np.min(loss,axis=1))
       
def viewResult(dataSizes,result,mode):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)): 
        plt.plot(dataSizes,result[mode[i]], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylabel('Loss Ratio', fontsize=14, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()

if __name__ == "__main__":
    main() 