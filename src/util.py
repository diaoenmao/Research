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

def to_numpy(x,ifcuda):
    x = x.data.cpu().numpy() if ifcuda else x.data.numpy()
    return x
    
def to_var(x,ifcuda,requires_grad=False):
    if torch.cuda.is_available() and ifcuda:
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)
        
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
    
def load_model(model, ifcuda, dir = './model/model.pth'):
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
        valid_indices = tuple(valid_indices)
        mesh_indices = np.meshgrid(*valid_indices, sparse=False, indexing='ij')  
        mesh_indices = tuple(mesh_indices)
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

# ===================Memory===================== 
def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

def gpuStats(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))
    
# ===================Function===================== 
def p_inverse(A):
    pinv = (A.t().matmul(A)).inverse().matmul(A.t())
    return pinv 
    
# ===================Figure=====================   
def showLoss(num_loss,setnames,TAG=''):
    plt.figure()
    for i in range(len(setnames)):
        loss_path = './output/{}/loss_{}.pkl'.format(setnames[i],TAG)
        loss_iter,loss_epoch = load(loss_path)
        if(setnames[i]=='train'):
            plt.plot(np.arange(0,num_loss),loss_epoch[:num_loss,],label=setnames[i],linewidth=1)
        elif(setnames[i]=='val'):
            plt.plot(np.arange(0,num_loss),loss_epoch[:num_loss,],label=setnames[i],linewidth=2)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return    
    