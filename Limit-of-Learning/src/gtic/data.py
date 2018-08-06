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
from torch.utils.data.sampler import SubsetRandomSampler
from util import *


seed = 1234

def fetch_dataset(data_name):
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
        test_dataset = datasets.MNIST(root=test_dir, train=False, download=True, transform=transform_test)

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
        test_dataset = datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
        
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
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        
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
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)
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
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)          
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
            test_dataset = datasets.SVHN(root=test_dir, split='test', download=True, transform=transform_test)
            
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
        test_dataset = datasets.EMNIST(root=test_dir, split=type, train=False, download=True, transform=transform_test)     
    return train_dataset,test_dataset

def split_dataset(train_dataset,test_dataset,data_size,batch_size,num_fold,p=0.8):
    indices = list(range(len(train_dataset)))
    data_idx = np.random.choice(indices, size=data_size, replace=False)
    if(num_fold==0):
        train_sampler = SubsetRandomSampler(data_idx)
        if(batch_size==0):
            batch_size = data_size
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, num_workers=2)    
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
            batch_size=1, num_workers=2)
        return train_loader,test_loader
    elif(num_fold==1):
        train_idx = np.random.choice(data_idx, size=int(data_size*p), replace=False)
        train_sampler = SubsetRandomSampler(train_idx)
        if(batch_size==0):
            batch_size = len(train_idx)
        train_loader = [torch.utils.data.DataLoader(dataset=train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, num_workers=2)]   
        validation_idx = list(set(data_idx) - set(train_idx))
        validation_sampler = SubsetRandomSampler(validation_idx)
        validation_loader = [torch.utils.data.DataLoader(dataset=train_dataset, 
                    batch_size=1, sampler=validation_sampler, num_workers=2)]
        return train_loader,validation_loader
    elif(num_fold>1 and num_fold<=len(indices)):
        splitted_idx = np.array_split(data_idx, num_fold)
        train_loader = []
        validation_loader = []
        for i in range(num_fold):
            validation_idx = splitted_idx[i]
            train_idx = list(set(data_idx) - set(validation_idx))
            train_sampler = SubsetRandomSampler(train_idx)
            if(batch_size==0):
                batch_size = len(train_idx)
            train_loader.append(torch.utils.data.DataLoader(dataset=train_dataset, 
                batch_size=batch_size, sampler=train_sampler, num_workers=2)) 
            validation_sampler = SubsetRandomSampler(validation_idx)
            validation_loader.append(torch.utils.data.DataLoader(dataset=train_dataset, 
                batch_size=1, sampler=validation_sampler, num_workers=2))
        return train_loader,validation_loader
    else:
        error("Invalid number of fold")
        exit()
    
def fetch_dataset_synth(input_feature,output_feature,high_dim=None,cov_mode='base',noise_sigma=np.sqrt(0.1),randomGen = np.random.RandomState(seed)):
    print('fetching data...')
    data_size = 50000
    test_size = 10000
    V = gen_cov_mat(input_feature,cov_mode)
    X = randomGen.multivariate_normal(np.zeros(input_feature),V,data_size+test_size)
    if(high_dim is None):
            beta = randomGen.randn(input_feature,output_feature)           
    else:
        if(high_dim>=input_feature):
            print('invalid high dimension')
            exit()
        valid_beta = randomGen.randn(high_dim,output_feature)
        empty_beta = np.zeros((input_feature-high_dim,output_feature))
        beta = np.vstack((valid_beta,empty_beta))
    mu = np.matmul(X,beta)
    eps = noise_sigma*randomGen.randn(*mu.shape)
    if(output_feature==1):
        y = mu + eps
    elif(output_feature>1):      
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
    X,y = X.astype(np.float32),y.astype(np.int64)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[:data_size,:]), torch.from_numpy(y[:data_size]))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[data_size:,:]), torch.from_numpy(y[data_size:]))
    return train_dataset,test_dataset
    
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

def get_mean_and_std(dataset,data_name=''):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(inputs.size(1)):
            mean[i] += inputs[:,i,].mean()
            std[i] += inputs[:,i,].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    save([mean,std],'./data/stats/stats_{}.pkl'.format(data_name))
    return mean, std
    
def extract_patches_2D(img,size):
    patches_fold_H = img.unfold(2, size[0], size[0])
    if(img.size(2) % size[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-size[0]:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, size[1], size[1])
    if(img.size(3) % size[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-size[1]:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(0,2,3,1,4,5).reshape(-1,img.size(1),size[0],size[1])
    return patches