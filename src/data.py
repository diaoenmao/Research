import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from util import *



mld_dataName = 'MNIST original'
mld_data_dir = './data/'
seed = 1234

def fetch_data_mld(dataSize=70000,valid_target=np.arange(10),randomGen = np.random.RandomState(seed)):
    print('fetching data...')
    data = datasets.fetch_mldata(mld_dataName, data_home=mld_data_dir)
    X = data.data
    y = data.target
    X,y = filter_data(X,y,valid_target)
    if(dataSize!=None):
        X,y = sample_data(dataSize,X,y,randomGen=randomGen)
    print('data ready')
    return X, y

def fetch_data_logistic(dataSize=1000,degree=100,coef=1.5,randomGen = np.random.RandomState(seed)):
    print('fetching data...') 
    X = randomGen.randn(dataSize,degree)
    beta = 10 / np.power(range(1,degree+1),coef)
    mu = X.dot(beta.reshape((degree,1)))
    y = randomGen.binomial(1, 1/(1 + np.exp(-mu)), size=None)  
    y = y.squeeze()
    print('data ready')
    return X, y

def fetch_data_circle(dataSize=1000,noise=0.1,factor=0.7,randomGen=np.random.RandomState(seed)):
    print('fetching data...') 
    X, y = datasets.make_circles(n_samples=dataSize, shuffle=False, noise=noise, random_state=randomGen, factor=factor)
    print('data ready')  
    return X, y
    
def sample_data(dataSize,X,y,randomGen = np.random.RandomState(seed)):
    if(dataSize<=X.shape[0]):
        sampled_X, sampled_y = shuffle(X, y, n_samples=dataSize, random_state=randomGen)
    else:
        print('sample size too large')
        exit()
    return sampled_X, sampled_y


def split_data_p(X,y,test_size=0.80,randomGen = np.random.RandomState(seed)):
    dataSize = X.shape[0]
    if(test_size>0 and test_size<1):
        test_size = np.int(dataSize*(1-test_size))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y,random_state=randomGen)
    return X_train, X_test, y_train, y_test

def split_data_holdout(X,y,test_size=0.75,randomGen = np.random.RandomState(seed)):
    X_train, X_val, y_train, y_val = split_data_p(X,y,test_size=test_size,randomGen = randomGen)
    return [X_train], [X_val], [y_train], [y_val]
    
def split_data_kfold(X,y,K,randomGen = np.random.RandomState(seed)):
    kfold = KFold(n_splits=K, random_state=randomGen)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for train_index, val_index in kfold.split(X, y):
        X_train.append(X[train_index,:])
        y_train.append(y[train_index])
        X_val.append(X[val_index,:])
        y_val.append(y[val_index])    
    return X_train, X_val, y_train, y_val
    
def split_data_loo(X,y,randomGen = np.random.RandomState(seed)):    
    dataSize = X.shape[0]
    X_train, X_val, y_train, y_val = split_data_kfold(X,y,dataSize,randomGen = randomGen)
    return X_train, X_val, y_train, y_val
    
def split_data_CrossValidation(X,y,K,test_size=0.75,randomGen = np.random.RandomState(seed)):
    if(K == 1):
        X_train, X_val, y_train, y_val  = split_data_holdout(X,y,test_size,randomGen = randomGen)
    elif(K < X.shape[0] and K > 1):
        X_train, X_val, y_train, y_val = split_data_kfold(X,y,K,randomGen = randomGen)
    elif(K == X.shape[0]):
        X_train, X_val, y_train, y_val = split_data_loo(X,y,randomGen = randomGen)
    else:
        print('Invalid K Fold')
        exit()  
    return X_train, X_val, y_train, y_val
    
def filter_data(X,y,valid_target):
    num_targets = valid_target.shape[0]
    valid_mask = np.isin(y,valid_target)
    X_filtered = X[valid_mask,:]
    y_filtered = y[valid_mask]
    for i in range(num_targets):
        y_filtered[y_filtered==valid_target[i]] = i
    return X_filtered, y_filtered

def gen_data_Linear(X,y,K,test_size,input_features,randomGen = np.random.RandomState(seed)):
    num_candidate_models = len(input_features)
    X_final_all, X_test_all, y_final, y_test = split_data_p(X,y,test_size=test_size,randomGen = randomGen)
    if(K == 'loo'):
        K = X_final_all.shape[0]
    X_train_CV, X_val_CV, y_train, y_val = split_data_CrossValidation(X_final_all,y_final,K,randomGen = randomGen)  
    X_train, X_val, X_final, X_test = [],[],[],[]
    for i in range(num_candidate_models):
        X_train_Model, X_val_Model = [],[]
        for k in range(K):
            X_train_Model.append(X_train_CV[k][:,input_features[i]])
            X_val_Model.append(X_val_CV[k][:,input_features[i]])
        X_train.append(X_train_Model)
        X_val.append(X_val_Model)
        X_final.append(X_final_all[:,input_features[i]])
        X_test.append(X_test_all[:,input_features[i]])
    return X_train, X_val, X_final, X_test, y_train, y_val, y_final, y_test

def gen_data_Full(X,y,K,test_size,num_candidate_models,randomGen = np.random.RandomState(seed)):
    X_final_all, X_test_all, y_final, y_test = split_data_p(X,y,test_size=test_size,randomGen = randomGen)
    X_train_CV, X_val_CV, y_train, y_val = split_data_CrossValidation(X_final_all,y_final,K,randomGen = randomGen)  
    X_train, X_val, X_final, X_test = [],[],[],[]
    for i in range(num_candidate_models):
        X_train_Model, X_val_Model = [],[]
        for k in range(K):
            X_train_Model.append(X_train_CV[k])
            X_val_Model.append(X_val_CV[k])
        X_train.append(X_train_Model)
        X_val.append(X_val_Model)
        X_final.append(X_final_all)
        X_test.append(X_test_all)
    return X_train, X_val, X_final, X_test, y_train, y_val, y_final, y_test
    
def get_data_stats(input,target=None,TAG=''):
    if input is not None:
        m_input = np.mean(input)
        std_input = np.std(input)
        print("input mean:%.3f, std:%.3f" % (m_input,std_input))
    else: 
        m_input = None
        std_input = None
    if target is not None:
        m_target = np.mean(target)
        std_target = np.std(target)
        print("target mean:%.3f, std:%.3f" % (m_target,std_target))
    else:
        m_target = None
        std_target = None
    save([m_input,std_input,m_target,std_target],'./data/stats/stats_{}.pkl'.format(TAG))
    return

def get_data_tensorset(input,target,input_datatype,target_datatype):
    input = torch.from_numpy(input).type(input_datatype)
    target = torch.from_numpy(target).type(target_datatype) 
    dataset = data_utils.TensorDataset(input,target)
    return dataset
    
def get_data_loader(input,target,input_datatype,target_datatype,batch_size):
    dataset = get_data_tensorset(input,target,input_datatype,target_datatype)
    data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
    
def normalize(input,target=None,input_datatype=torch.FloatTensor,target_datatype=torch.FloatTensor,TAG=''):
    m_input,std_input,m_target,std_target = load('./data/stats/stats_{}.pkl'.format(TAG))
    if input is not None:
        norm_input = (input.numpy()-m_input)/std_input
        norm_input = torch.FloatTensor(norm_input).type(input_datatype)
    else:
        norm_input = None
    if target is not None:
        norm_target = (target.numpy()-m_target)/std_target
        norm_target = torch.FloatTensor(norm_target).type(target_datatype)
    else:
        norm_target = None
    return norm_input,norm_target

def denormalize(norm_input,norm_target=None,input_datatype=torch.FloatTensor,target_datatype=torch.FloatTensor,TAG=''):
    m_input,std_input,m_target,std_target = load('./data/stats/stats_{}.pkl'.format(TAG))
    if norm_input is not None:
        denorm_inputs = norm_input.numpy()*std_input+m_input
        input = torch.FloatTensor(denorm_inputs).type(input_datatype)
    else:
        input = None
    if norm_target is not None:
        denorm_targets = norm_target.numpy()*std_target+m_target   
        target = torch.FloatTensor(denorm_targets).type(target_datatype)
    else:
        target = None
    return input,target