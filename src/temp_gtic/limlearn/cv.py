import numpy as np
from sklearn.model_selection import train_test_split,KFold

def holdout(dataX,dataY,p):
    dataX_train, dataX_validation, dataY_train, dataY_validation = train_test_split(dataX, dataY, test_size=(1-p))
    return (dataX_train, dataX_validation, dataY_train, dataY_validation)
    
def kfold(dataX,dataY,k):
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
    
def loo(dataX,dataY):  
    dataSize = dataX.shape[0]
    dataX_train, dataX_validation, dataY_train, dataY_validation = kfold(dataX,dataY,dataSize)
    return (dataX_train, dataX_validation, dataY_train, dataY_validation)   