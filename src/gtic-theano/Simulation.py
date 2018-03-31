import numpy
import theano
import theano.tensor as T
import pickle
import sys
import time
import os.path
from MLP import *
from TIC import *
from CV import *
from Data import *
from Train import *

def runExperiment(dataSizes,num_hidden_nodes_list):    
    oracleSize = 10000
    ifload = True
    models_dir = './model/models.pkl'
    TIC_dir = './model/TIC.pkl'
   
    if(os.path.exists(models_dir) and ifload):
        models = loadModel(models_dir) 
    else:
        models=[]
        for i in num_hidden_nodes_list:
            models.append(build_sk_mlp(num_hidden_nodes_list[i]))
        saveModel(models,models_dir)
        
    models_CV=[]   
    for i in num_hidden_nodes_list:
            models_CV.append(build_sk_mlp(num_hidden_nodes_list[i],False))
            
    max_num_hidden_layer = 1
    num_hidden_layer_list = []
    for i in num_hidden_nodes_list:
        num_hidden_layer_list.append(len(num_hidden_nodes_list[i]))
    max_num_hidden_layer = max(num_hidden_layer_list)
    num_hidden_layers = list(range(1,max_num_hidden_layer)) 
    
    if(os.path.exists(TIC_dir) and ifload):
        tensorslist,num_hidden_layer_list = loadModel(TIC_dir) 
    else:
        tensorslist = BuildModelClass(num_hidden_layers)
        saveModel([tensorslist,num_hidden_layer_list],TIC_dir)
    
    tics = []
    for i in range(len(tensorslist)):   
        tensors = tensorslist[i]
        input = [tensors[-2],tensors[-1]]
        tic = TIC(*input).tic()     
        tics.append(theano.function(inputs=tensors[:-2],outputs=tic))
                  
    oracleX,oracleY = generate_data(oracleSize)   
    required_tics = [tics[i] for i in num_hidden_layer_list]
    loss_oracle = np.zeros((len(models),len(dataSizes)))
    model_selected = {'TIC': [], \
                'CV_holdout': [], 'CV_3fold': [], 'CV_10fold': [], 'CV_loo': []}
    for i in range(len(dataSizes)):
        dataX,dataY = generate_data(dataSizes[i])
        #oracleX,oracleY = generate_data(oracleSize)
        for j in range(len(models)):
            models[j] = train_sk_mlp(dataX,dataY,models[j])
            loss_oracle[j,i] = loss_sk_mlp(oracleX,oracleY,models[j])    
        model_selected['TIC'].append(runTIC(dataX,dataY,models,tics))
        model_selected['CV_holdout'].append(runCV_holdout(dataX,dataY,models))
        model_selected['CV_3fold'].append(runCV_kfold(dataX,dataY,3,models))
        model_selected['CV_10fold'].append(runCV_kfold(dataX,dataY,10,models))
        model_selected['CV_loo'].append(runCV_loo(dataX,dataY,models)    
        
def runTIC(dataX,dataY,models,tics)
    lossAndTIC = np.zeros(len(models))
    for i in range(len(models)):
        input = [dataX, dataY]
        param = param_sk_mlp(models[i])
        input.extend(param)  
        lossAndTIC[i] = loss_sk_mlp(dataX,dataY,models[i]) + tics[i](*input)
    return np.argmin(lossAndTIC)
    
def runCV_holdout(dataX,dataY,models):
    loss = np.zeros(len(models))
    dataX_train, dataX_validation, dataY_train, dataY_validation=CV_holdout(dataX,dataY, 0.7)
    for i in range(len(models)):
        models[i].fit(dataX_train, dataY_train)
        loss = loss_sk_mlp(dataX_validation,dataY_validation,models[i])
    return np.argmin(loss)

def runCV_kfold(dataX,dataY,k,models):
    loss = np.zeros((len(models),len(dataX_train)))
    dataX_train, dataX_validation, dataY_train, dataY_validation=CV_kfold(dataX,dataY,k)
    for i in range(len(models)):
        for j in range(len(dataX_train)):
            models[i].fit(dataX_train[j], dataY_train[j])
            loss[i,j]=loss_sk_mlp(dataX_validation[j], dataY_validation[j],models[i])
    return np.argmin(np.min(loss,axis=1))

def runCV_loo(dataX,dataY,models):
    loss = np.zeros((len(models),len(dataX_train)))
    dataX_train, dataX_validation, dataY_train, dataY_validation=CV_loo(dataX,dataY)
    for i in range(len(models)):
        for j in range(len(dataX_train)):
            models[i].fit(dataX_train[j], dataY_train[j])
            loss[i,j]=loss_sk_mlp(dataX_validation[j], dataY_validation[j],models[i])
    return np.argmin(np.min(loss,axis=1))
       

if __name__ == "__main__":
    main() 