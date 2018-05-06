import pickle
import time
import torch
import itertools
import os
import gc
import sys
import psutil
import shutil
import zipfile
import numpy as np
import seaborn as sns
from torch.autograd import Variable
from matplotlib import pyplot as plt

def save(input,dir,protocol = 3):
    dirname = os.path.dirname(dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    pickle.dump(input, open(dir, "wb" ), protocol=protocol)
    return

def load(dir):
    return pickle.load(open(dir, "rb" ))        

def save_model(model, dir = './model/model.pth'):
    dirname = os.path.dirname(dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), dir)
    return
    
def load_model(model, dir = './model/model.pth'):
    checkpoint = torch.load(dir)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:       
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def print_model(model):
    for p in model.parameters():
        print(p)
    return

def zip_dir(paths,zip_name):
    dirname = os.path.dirname(zip_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for i in range(len(paths)):
        path = paths[i]
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))
    zipf.close()
    return
        
def remove_dir(paths):
    for i in range(len(paths)):
        path = paths[i]
        if os.path.exists(path):
            shutil.rmtree(path)
    return 
    
def get_correct_cnt(output,target):
    max_index = output.max(dim = 1)[1]  
    correct_cnt = (max_index == target).float().sum()
    return correct_cnt
    
def get_acc(output,target):
    max_index = output.max(dim = 1)[1]
    correct_cnt = get_correct_cnt(output,target)
    total_cnt = output.size()[0]
    acc = correct_cnt/total_cnt
    return acc

def get_data_dist(data):
    plt.figure()
    plt.hist(data)
    plt.show()

def gen_input_features(dims,init_size=None,step_size=None,start_point=None):
    if (init_size is None):
        init_size=[2]*len(dims) 
    if (step_size is None):
        step_size = [1]*len(dims) 
    if (start_point is None):
        start_point = [np.int(dims[i]/2) for i in range(len(dims))] 
    ifcovered = [False]*len(dims)
    input_features = []
    j = 0
    while(not np.all(ifcovered)):
        valid_indices = []
        for i in range(len(dims)):
            indices = np.arange(start_point[i]-init_size[i]/2-j*step_size[i],start_point[i]+init_size[i]/2+j*step_size[i],dtype=np.int)
            cur_valid_indices = indices[(indices>=0)&(indices<=(dims[i]-1))]
            ifcovered[i] = np.any(indices<=0) and np.any(indices>=(dims[i]-1))
            valid_indices.append(cur_valid_indices)
        mesh_indices = tuple(np.meshgrid(*valid_indices, sparse=False, indexing='ij'))
        raveled_indices = np.ravel_multi_index(mesh_indices, dims=dims, order='C') 
        raveled_indices = raveled_indices.ravel()    
        input_features.append(raveled_indices)
        j = j + 1
    return input_features

def gen_hidden_layers(max_num_nodes,init_size=None,step_size=None):
    if (init_size is None):
        init_size=[1]*len(max_num_nodes) 
    if (step_size is None):
        step_size = [1]*len(max_num_nodes)
    num_nodes = []
    hidden_layers = []
    for i in range(len(max_num_nodes)):
        num_nodes.append(list(range(init_size[i],max_num_nodes[i]+1,step_size[i])))
    while(len(num_nodes) != 0):
        hidden_layers.extend(list(itertools.product(*num_nodes)))
        del num_nodes[-1]   
    return hidden_layers
    
# ===================Function===================== 
def p_inverse(A):
    pinv = (A.t().matmul(A)).inverse().matmul(A.t())
    return pinv 

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x
    
# ===================Figure=====================   
def showLoss(setnames,TAG=''):
    plt.figure()
    for i in range(len(setnames)):
        loss_path = './output/{}/loss_acc_{}.pkl'.format(setnames[i],TAG)
        loss_iter,loss_epoch,regularized_loss_iter,regularized_loss_epoch,acc_iter,acc_epoch = load(loss_path)
        num_loss_iter = len(loss_iter)
        num_loss_epoch = len(loss_epoch)
        if(setnames[i]=='train'):
            plt.plot(np.arange(0,num_loss_iter),loss_iter,label='loss_'+setnames[i],linewidth=1)
            plt.plot(np.arange(0,num_loss_iter),regularized_loss_iter,label='regularized_loss_'+setnames[i],linewidth=1)
        elif(setnames[i]=='val'):
            plt.plot(np.arange(0,num_loss_iter),loss_iter,label='loss_'+setnames[i],linewidth=2)
            plt.plot(np.arange(0,num_loss_iter),regularized_loss_iter,label='regularized_loss_'+setnames[i],linewidth=2)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return
    
def show(loss,setnames):
    plt.figure()
    for i in range(len(setnames)):
        num_loss = len(loss[i])
        plt.plot(np.arange(0,num_loss),loss[i],label=setnames[i],linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()    
    return
    