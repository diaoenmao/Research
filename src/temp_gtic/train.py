import numpy as np
from sklearn.neural_network import MLPClassifier

def build_sk_mlp(num_hidden_nodes,warm_start=True,sovler='lbfgs'):
    mlp = MLPClassifier(hidden_layer_sizes=tuple(num_hidden_nodes),activation='tanh',solver=sovler,warm_start=warm_start)
    return mlp
    
def train_sk_mlp(dataX,dataY,mlp):
    mlp.fit(dataX,dataY)
    return mlp
    
def loss_sk_mlp(dataX,dataY,mlp):
    P = mlp.predict_proba(dataX)
    # print(P.shape)
    # print(dataY.shape)
    n = dataX.shape[0]
    #print(np.max(P[np.arange(n), dataY]))
    loglik=np.mean(np.log(P[np.arange(n), dataY]))
    return -loglik

def error_sk_mlp(dataX,dataY,mlp):
    predictY = mlp.predict(dataX)
    error_rate = np.mean(predictY!=dataY)
    return error_rate
    
def param_sk_mlp(mlp):
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    param = []
    #print(coefs[-1].shape)
    #print(intercepts[-1].shape)
    for i in range(len(coefs)):
        param.append(np.concatenate((intercepts[i].reshape(1,coefs[i].shape[1]),coefs[i]),axis=0))
    return param