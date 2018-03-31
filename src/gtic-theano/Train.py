import numpy as np
from sklearn.neural_network import MLPClassifier

def build_sk_mlp(num_hidden_nodes,warm_start=True):
    mlp = MLPClassifier(hidden_layer_sizes=tuple(num_hidden_nodes),activation='tanh',solver='lbfgs',warm_start=warm_start)
    return mlp
    
def train_sk_mlp(dataX,dataY,mlp)   
    mlp.fit(dataX,dataY)
    return mlp
    
def loss_sk_mlp(dataX,dataY,mlp)
    P = mlp.predict_proba(dataX)
    n = dataX.shape[0]
    loglik=np.mean(np.log(P[np.arange(n), dataY]))
    return -loglik


def param_sk_mlp(mlp)
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    param = []
    for i in range(len(coefs)):
        param.append(np.concatenate(intercepts[i],coef[i],axis=0))
    return param