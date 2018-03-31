import numpy
import theano
import theano.tensor as T
import limlearn as lol
import limlearn.cv,limlearn.mlp

import sys
import time
import os.path
from itertools import compress

from data import *
from train import *
from sl import *

def runExperiment(dataSizes,num_hidden_nodes_list, mode):
    max_dataSize = max(dataSizes)
    oracleSize = 10000
    totalSize = max_dataSize+oracleSize
    ifload = False
    models_dir = './model/models_mnist.pkl'
    TIC_dir = './model/tic.pkl'
    if(os.path.exists(models_dir) and ifload):
        models = load(models_dir) 
    else:
        models=[]
        for i in range(len(num_hidden_nodes_list)):
            models.append(build_sk_mlp(num_hidden_nodes_list[i],False))
        save(models,models_dir)
        
    models_cv=[]   
    for i in range(len(num_hidden_nodes_list)):
            models_cv.append(build_sk_mlp(num_hidden_nodes_list[i],False))
            
    num_hidden_layer_list = np.zeros(len(num_hidden_nodes_list))
    for i in range(len(num_hidden_nodes_list)):
        num_hidden_layer_list[i] = len(num_hidden_nodes_list[i])
    
    num_hidden_layers,indices = np.unique(num_hidden_layer_list,return_inverse=True) 
    
    if(os.path.exists(TIC_dir) and ifload):
        tensorslist,num_hidden_layer_list,indices = load(TIC_dir) 
    else:
        tensorslist = limlearn.mlp.BuildModelClass(np.int32(num_hidden_layers).tolist())
        save([tensorslist,num_hidden_layer_list,indices],TIC_dir)
    
    tics = []
    for i in range(len(tensorslist)):   
        tensors = tensorslist[i]
        input = [tensors[-2],tensors[-1]]
        tic = lol.tic(*input)     
        tics.append(theano.function(inputs=tensors[:-2],outputs=tic))
    
    n_components = 4
    #select_num = np.array([0,1])
    select_num = np.array([0,1])
    #select_num = np.array([0,1,2,3,4,5,6,7,8,9])
    totalX,totalY = generate_mnist_pca_data(totalSize,n_components,select_num)
    print(np.unique(totalY))
    oracleX,oracleY = totalX[-oracleSize:,:],totalY[-oracleSize:]
    required_tics = [tics[i] for i in indices]
    loss_oracle = np.zeros((len(models),len(dataSizes)))
    error_rate = np.zeros((len(models),len(dataSizes)))
    timing = {k: [] for k in mode}
    model_selected = {k: [] for k in mode}
    
    dataX_all,dataY_all = totalX[:max_dataSize,:],totalY[:max_dataSize]
    data_store = []
    oracle = [oracleX,oracleY]
    for i in range(len(dataSizes)):
        print(dataSizes[i])
        
        
        dataX = dataX_all[:dataSizes[i],:]
        dataY = dataY_all[:dataSizes[i]]
        
        perm = np.random.permutation(dataX.shape[0])
        dataX = dataX[perm]
        dataY = dataY[perm]
    
        data_store.append([dataX,dataY])   
        
        thresh = np.floor((np.sqrt(dataSizes[i])/(2+1))-1).astype(np.int)
        #print(thresh)
        active = np.squeeze(num_hidden_nodes_list<=thresh)
        active_models = list(compress(models, active))
        active_models_cv = list(compress(models_cv, active))
        
        #active_models = models
        active_models_cv = models_cv
        
        tmp_time = np.zeros(len(models))      
        for j in range(len(models)):
            start=time.time()
            models[j] = train_sk_mlp(dataX,dataY,models[j])
            end=time.time()
            tmp_time[j] = end-start
        print('train done')
        

        start=time.time()
        model_selected['tic'].append(run_tic(dataX,dataY,active_models,required_tics))
        end=time.time()
        timing['tic'].append(np.sum(tmp_time)+end-start)       
        print('tic done')
        
        start=time.time()
        model_selected['holdout'].append(run_holdout(dataX,dataY,active_models_cv))
        end=time.time()
        timing['holdout'].append(tmp_time[model_selected['holdout'][-1]]+end-start)
        print('holdout done') 
        
        start=time.time()
        model_selected['3fold'].append(run_kfold(dataX,dataY,3,active_models_cv))
        end=time.time()
        timing['3fold'].append(tmp_time[model_selected['3fold'][-1]]+end-start)
        print('3fold done') 
        
        start=time.time()
        model_selected['10fold'].append(run_kfold(dataX,dataY,10,active_models_cv))
        end=time.time()
        timing['10fold'].append(tmp_time[model_selected['10fold'][-1]]+end-start)
        print('10fold done')
        print('cv done') 
        
        # start=time.time()
        # model_selected['loo'].append(run_loo(dataX,dataY,active_models_cv))
        # end=time.time()
        # timing['loo'].append(tmp_time[model_selected['loo'][-1]]+end-start)

        for j in range(len(models)):
            loss_oracle[j,i] = loss_sk_mlp(oracleX,oracleY,models[j])
            error_rate[j,i] = error_sk_mlp(oracleX,oracleY,models[j])
            
        print([error_rate[model_selected['tic'][-1],i],error_rate[model_selected['holdout'][-1],i],error_rate[model_selected['3fold'][-1],i],error_rate[model_selected['10fold'][-1],i]])
        print(error_rate[:,i])
        print([model_selected['tic'],model_selected['holdout'],model_selected['3fold'],model_selected['10fold']])
        print([timing['tic'],timing['holdout'],timing['3fold'],timing['10fold']])
    return data_store,oracle,loss_oracle,error_rate,model_selected,timing
    
    
def main():
    start=time.time()
    result_dir = 'result_mnist.pkl'
    init_dataSize = 200
    batch_dataSize = 10
    end_dataSize = 1000
    dataSizes = range(init_dataSize,end_dataSize+1,batch_dataSize)
    mode = ['tic', 'holdout', '3fold', '10fold']
    num_hidden_nodes_list = get_num_hidden_nodes()
    data_store,oracle,loss_oracle,error_rate,model_selected,timing = runExperiment(dataSizes, num_hidden_nodes_list, mode)
    loss_ratio = getLossRatio(loss_oracle,model_selected,mode)
    print(loss_ratio)
    save([data_store,oracle,dataSizes,loss_oracle,error_rate,model_selected,timing,loss_ratio,mode],result_dir) 
    end=time.time()
    print('total time: %d' % (end-start))

def run_tic(dataX,dataY,models,tics):
    lossAndTIC = np.zeros(len(models))
    for i in range(len(models)):
        print(i)
        input = [dataX, dataY]
        param = param_sk_mlp(models[i])
        num_param = 0
        for j in range(len(param)):
            num_param = num_param + (param[j].shape[0]+1)*param[j].shape[1]
        input.extend(param)
        loss = loss_sk_mlp(dataX,dataY,models[i])
        TIC = tics[i](*input)
        AIC = num_param/dataX.shape[0]
        if(TIC<0):
            TIC = 100000
        print(loss)
        print(TIC)
        print(AIC)
        lossAndTIC[i] = loss_sk_mlp(dataX,dataY,models[i]) + TIC
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
    return np.argmin(np.mean(loss,axis=1))

def run_loo(dataX,dataY,models):
    dataX_train, dataX_validation, dataY_train, dataY_validation=lol.cv.loo(dataX,dataY)
    loss = np.zeros((len(models),len(dataX_train)))
    for i in range(len(models)):
        for j in range(len(dataX_train)):
            models[i].fit(dataX_train[j], dataY_train[j])
            loss[i,j]=loss_sk_mlp(dataX_validation[j], dataY_validation[j],models[i])
    return np.argmin(np.mean(loss,axis=1))
       
def getLossRatio(loss_oracle,model_selected,mode):
    opt_loss_oracle = np.min(loss_oracle,axis=0)
    loss_ratio = {k: [] for k in mode}
    ave_loss_ratio = []
    for i in range(len(mode)):
        loss_ratio[mode[i]] = loss_oracle[model_selected[mode[i]],np.arange(loss_oracle.shape[1])]/opt_loss_oracle
        #loss_ratio[mode[i]][[loss_ratio[mode[i]]>1.5]] = np.max(loss_ratio[mode[i]][loss_ratio[mode[i]]<=1.5])
        loss_ratio[mode[i]][[loss_ratio[mode[i]]==np.inf]] = np.max(loss_ratio[mode[i]][loss_ratio[mode[i]]!=np.inf])
        ave_loss_ratio.append(np.mean(loss_ratio[mode[i]]))
    return loss_ratio

def get_num_hidden_nodes():
    #num_hidden_nodes_list = [[300]]
    num_hidden_nodes_list = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    #num_hidden_nodes_list = [[300,20],[500,20],[800,20],[1000,20]]
    #num_hidden_nodes_list = [[300,10],[500,10],[800,10],[1000,10]]
    #num_hidden_nodes_list = [[300],[500],[800],[1000],[300,100],[500,150],[500,300],[800,500],[1000,500]]
    #num_hidden_nodes_list = [[300],[500],[800],[1000],[300,100],[500,150],[500,300],[800,500],[1000,500],[1500,1000,500],[2000,1500,1000,500],[2500, 2000, 1500, 1000, 500]]
    return num_hidden_nodes_list
    
if __name__ == "__main__":
    main() 