import numpy as np
from sklearn.model_selection import train_test_split

def CV_holdout(dataX,dataY,p):
    dataX_train, dataX_validation, dataY_train, dataY_validation = train_validation_split(dataX, dataY, validation_size=(1-p))
    return (dataX_train, dataX_validation, dataY_train, dataY_validation)
    
def CV_kfold(dataX,dataY,k):
    kfold = KFold(n_splits=k)
    dataX_train = []
    dataX_validation = []
    dataY_train = []
    dataY_validation = []
    for train_index, validation_index in kfold.split(dataX, dataY):
        dataX_train.append(dataX[train_index,:])               
        dataX_validation.append(dataX[validation_index,:])
        dataY_train.append(dataY[train_index])
        dataY_validation.append(dataY[validation_index])    
    return (dataX_train, dataX_validation, dataY_train, dataY_validation)
    
def CV_loo(dataX,dataY):  
    dataSize = dataX.shape[0]
    CV_kfold(dataX,dataY,dataSize)
    return (dataX_train, dataX_validation, dataY_train, dataY_validation)   