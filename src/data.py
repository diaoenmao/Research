import numpy as np
import torch
import os
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from util import *


seed = 1234

def fetch_data(data_name,batch_size):
    print('fetching data...')
    stats_name = './data/stats/stats_{}.pkl'.format(data_name)
    if(data_name=='MNIST'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/test/'.format(data_name)
        if(not os.path.exists(train_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(not os.path.exists(test_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(os.path.exists(stats_name)):
            mean,std = load(stats_name)
        else:
            train_dataset = datasets.MNIST(root=train_dir, train=True, download=True, transform=transforms.ToTensor())
            mean,std = get_mean_and_std(train_dataset,data_name)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.MNIST(root=train_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_dataset = datasets.MNIST(root=test_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif(data_name=='CIFAR10' or data_name=='CIFAR100'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/test/'.format(data_name)
        if(not os.path.exists(train_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(not os.path.exists(test_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(os.path.exists(stats_name)):
            mean,std = load(stats_name)
        else:
            train_dataset = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transforms.ToTensor())
            mean,std = get_mean_and_std(train_dataset,data_name)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_dataset = datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    elif(data_name=='Imagenet-12'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/test/'.format(data_name)
        if(not os.path.exists(train_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(not os.path.exists(test_dir)):
            os.makedirs(train_dir, exist_ok=True)
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)      
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    elif(data_name=='SVHN_train' or data_name=='SVHN_extra' or data_name=='SVHN_all'):
        head_data_name,type = data_name.split('_')
        train_dir = './data/{}/train/'.format(head_data_name)
        test_dir = './data/{}/test/'.format(head_data_name)
        if(not os.path.exists(train_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(not os.path.exists(test_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(type=='train'):
            if(os.path.exists(stats_name)):
                mean,std = load(stats_name)
            else:
                train_dataset = datasets.SVHN(root=train_dir, split='train', download=True, transform=transforms.ToTensor())
                mean,std = get_mean_and_std(train_dataset,data_name)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            train_dataset = datasets.SVHN(root=train_dir, split='train', download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        elif(type=='extra'):  
            if(os.path.exists(stats_name)):
                mean,std = load(stats_name)
            else:
                train_dataset = datasets.SVHN(root=train_dir, split='extra', download=True, transform=transforms.ToTensor())
                mean,std = get_mean_and_std(train_dataset,data_name)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            train_dataset = datasets.SVHN(root=train_dir, split='extra', download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)            
        elif(type=='all'):
            if(os.path.exists(stats_name)):
                mean,std = load(stats_name)
            else:
                train_train_dataset = datasets.SVHN(root=train_dir, split='train', download=True, transform=transforms.ToTensor())
                extra_train_dataset = datasets.SVHN(root=train_dir, split='extra', download=True, transform=transforms.ToTensor())
                train_dataset = data_utils.ConcatDataset([train_train_dataset,extra_train_dataset])
                mean,std = get_mean_and_std(train_dataset,data_name)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            train_train_dataset = datasets.SVHN(root=train_dir, split='train', download=True, transform=transform_train)
            extra_train_dataset = datasets.SVHN(root=train_dir, split='extra', download=True, transform=transform_train)
            train_dataset = data_utils.ConcatDataset([train_train_dataset,extra_train_dataset])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            
    elif(data_name=='EMNIST_byclass' or data_name=='EMNIST_bymerge' or data_name=='EMNIST_balanced' or data_name=='EMNIST_letters' or data_name=='EMNIST_digits' or data_name=='EMNIST_mnist'):
        head_data_name,type = data_name.split('_')
        train_dir = './data/{}/train/'.format(head_data_name)
        test_dir = './data/{}/test/'.format(head_data_name)
        if(not os.path.exists(train_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(not os.path.exists(test_dir)):
            os.makedirs(train_dir, exist_ok=True)
        if(os.path.exists(stats_name)):
            mean,std = load(stats_name)
        else:
            train_dataset = datasets.EMNIST(root=train_dir, split=type, download=True, transform=transforms.ToTensor())
            mean,std = get_mean_and_std(train_dataset,data_name)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.EMNIST(root=train_dir, split=type, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_dataset = datasets.EMNIST(root=test_dir, split=type, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)        
    return train_loader,test_loader
    
def fetch_data_linear(dataSize,input_features,out_features=1,high_dim=None,cov_mode='base',noise_sigma=np.sqrt(0.1),randomGen = np.random.RandomState(seed)):
    print('fetching data...')
    V = gen_cov_mat(input_features,cov_mode)
    X = randomGen.multivariate_normal(np.zeros(input_features),V,dataSize)
    if(high_dim is None):
            beta = randomGen.randn(input_features,out_features)           
    else:
        if(high_dim>=input_features):
            print('invalid high dimension')
            exit()
        valid_beta = randomGen.randn(high_dim,out_features)
        empty_beta = np.zeros((input_features-high_dim,out_features))
        beta = np.vstack((valid_beta,empty_beta))
    mu = np.matmul(X,beta)
    eps = noise_sigma*randomGen.randn(*mu.shape)
    if(out_features==1):
        y = mu + eps
    elif(out_features>1):      
        p = softmax(mu + eps)
        y = []
        for i in range(X.shape[0]):
            sample = randomGen.multinomial(1,p[i,])
            y.append(np.where(sample==1)[0][0])
        y = np.array(y)
    else:
        print('invalid dimension')
        exit()
    print('data ready')
    return X, y
    
def gen_cov_mat(dim,mode,zo=0.5):
    if(mode=='base'):
        V = np.eye(dim)
    elif(mode=='corr'):
        V = np.full((dim, dim), zo)
        V = V + (1-zo)*np.eye(dim)
    elif(mode=='decay_corr'):
        indices = np.arange(dim)
        valid_indices = [indices,indices]
        mesh_indices = np.meshgrid(*valid_indices, sparse=False, indexing='ij')
        exponent = np.abs(mesh_indices[0]-mesh_indices[1])
        V = np.power(zo,exponent)
    else:
        print('invalid covariance mode')
        exit()
    return V

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

def split_data_p(X,y,test_size=0.2,randomGen = np.random.RandomState(seed)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomGen)
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

def get_data_tensorset(input,target,input_datatype,target_datatype,device):
    input = torch.from_numpy(input).type(input_datatype).to(device)
    target = torch.from_numpy(target).type(target_datatype).to(device)
    dataset = data_utils.TensorDataset(input,target)
    return dataset
    
def get_data_loader(input,target,input_datatype,target_datatype,device,batch_size):
    dataset = get_data_tensorset(input,target,input_datatype,target_datatype,device)
    data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
    
def normalize(input,target=None,TAG=''):
    m_input,std_input,m_target,std_target = load('./data/stats/stats_{}.pkl'.format(TAG))
    if input is not None:
        norm_input = (input-m_input)/std_input
    else:
        norm_input = None
    if target is not None:
        norm_target = (target.numpy()-m_target)/std_target
    else:
        norm_target = None
    return norm_input,norm_target

def denormalize(norm_input,norm_target=None,TAG=''):
    m_input,std_input,m_target,std_target = load('./data/stats/stats_{}.pkl'.format(TAG))
    if norm_input is not None:
        denorm_input = norm_input*std_input+m_input
    else:
        denorm_input = None
    if norm_target is not None:
        denorm_target = norm_target*std_target+m_target   
    else:
        denorm_target = None
    return denorm_input,denorm_target

def get_mean_and_std(dataset,data_name=''):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(inputs.size(1)):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    save([mean,std],'./data/stats/stats_{}.pkl'.format(data_name))
    return mean, std
