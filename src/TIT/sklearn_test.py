import numpy as np
from MLP import *
import theano

def generateData(dataSize):
    dataX, dataY = datasets.make_circles(n_samples=dataSize, shuffle=True, noise=0.2, random_state=None, factor=0.5)
    dataY = dataY.reshape((-1,1))
    return dataX,dataY

def fit(dataX, dataY, mlp):
    mlp.fit(dataX, dataY)
    return mlp

def getLoss(dataX, dataY, mlp, flag='CrossEntropy'):
    if(flag == 'CrossEntropy'):
        P = mlp.predict_proba(dataX)
        n = dataX.shape[0]
        loss=-np.mean(np.log(P[np.arange(n), dataY]))
    return loss
    
def getParam(mlp):
    W = mlp.coefs_
    b = mlp.intercepts_
    param=[]
    for i in range(len(beta)):
        cur_W = W[i]
        cur_b = b[i]
        cur_DM = np.concatenate(cur_b.reshape((1,cur_b.shape[0])), cur_W, axis=0)
        cur_param_num = cur_DM.shape[0]*cur_DM.shape[1]
        cur_param = cur_DM.reshape((cur_param_num,)).tolist()
        param.extend(cur_param)
    param = np.array(param)
    return param
    
def getTIC(dataX, dataY, model, param):
    TIC_model = theano.function(inputs=[model[0],model[1],model[2]],outputs=model[3])
    return TIC_model

def runExperiment(dataSize_init, dataSize_end, batchSize, architecture):
    dataSizes = range(dataSize_init,dataSize_end+1,batchSize)
    t = len(dataSizes)
    dataX = []
    dataY = []
    for s in dataSizes:
        tmpX,tmpY=generateData(s)
        dataX.append(tmpX)
        dataY.append(tmpY)
    for a in architecture:
        
        for a in architecture:
            hiddenlayerSize = len(a[1])
            mlp = MLPClassifier(hidden_layer_sizes=tuple(a[1]),activation='tanh',solver='lbfgs')
def main():
    runExperiment()
    
    
