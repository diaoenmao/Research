import time
import torch
import torchvision
import itertools
import os
import sys
import shutil
import zipfile
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchsummary import summary

def save(input,dir,protocol = 2,mode='torch'):
    dirname = os.path.dirname(dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if(mode=='torch'):
        torch.save(input,dir,pickle_protocol=protocol)
    elif(mode=='numpy'):
        np.save(dir,input)
    else:
        error('Not supported save mode')
    return

def load(dir,mode='torch'):
    if(mode=='torch'):
        return torch.load(dir)
    elif(mode=='numpy'):
        return np.load(dir)  
    else:
        error('Not supported save mode')
    return                

def save_model(model, dir):
    dirname = os.path.dirname(dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), dir)
    return
    
def load_model(model, dir):
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
        
def remove_dir(paths):
    for i in range(len(paths)):
        path = paths[i]
        if os.path.exists(path):
            shutil.rmtree(path)
    return 

def filenames_in(dir,target_ext):
    filenames_ext = os.listdir(dir)
    filenames = []
    for filename_ext in filenames_ext:
        filename,ext = filename_ext.rsplit('.',1)
        if(ext == target_ext):
            filenames.append(filename)
    filenames.sort()
    return filenames
    
def get_correct_cnt(output,target):
    max_index = output.max(dim = 1)[1]  
    correct_cnt = (max_index == target).float().sum()
    return correct_cnt
    
def get_acc(output,target,topk):  
    with torch.no_grad():
        maxk = min(max(topk),output.size(1))
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            if(k<=maxk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                cur_acc = correct_k.mul_(100.0 / batch_size)
                acc.append(cur_acc.item())
            else:
                acc.append(1.0)
    return acc

def modelselect_input_feature(dims,init_size=2,step_size=1,start_point=None):
    if(isinstance(dims, int)):
        dims = [dims]
    init_size=[init_size]*len(dims) 
    step_size = [step_size]*len(dims) 
    if (start_point is None):
        start_point = [np.int(dims[i]/2) for i in range(len(dims))]
    else:
        start_point = [0 for i in range(len(dims))]
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

def print_result(epoch,train_result,test_result):
    print('Test Epoch: {0}\tPatch Size: {patch_size.sum}\tLoss: {losses.avg:.4f}\tPSNR: {psnrs.avg:.4f}\tTime: {time.sum}'
        .format(epoch,patch_size=train_result[2],losses=test_result[2],psnrs=test_result[3],time=train_result[0]))
    return

def merge_result(train_result,test_result,new_train_result,new_test_result):
    if(train_result is None):
        train_result = list(new_train_result)
        test_result = list(new_test_result)            
    else:
        for i in range(len(train_result)):
            train_result[i].merge(new_train_result[i])
            test_result[i].merge(new_test_result[i])
    return train_result,test_result

def PIL_to_CV2(pil_img):
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img
    
def CV2_to_PIL():
    cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img

def to_img(x):
    x = x.clamp(0, 1)
    return x
    
def save_img(img,nrow,path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    gridded_img = make_grid(img,nrow=nrow)
    save_image(gridded_img, path)
            
# ===================Object===================== 

class Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history_val = []
        self.history_avg = [0]
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history_val.append(self.val)
        self.history_avg[0] = self.avg
        
    def merge(self, meter):
        self.history_val.extend(meter.history_val)
        self.history_avg.extend(meter.history_avg)
    
# ===================Function===================== 
def p_inverse(A):
    pinv = (A.t().matmul(A)).inverse().matmul(A.t())
    return pinv 

def RGB_to_YCbCr(input):
    input = input*255
    output = input.new_empty(input.size())
    if(input.dim()==3):
        output[0, :, :] = input[0, :, :] * 0.299 + input[1, :, :] * 0.587 + input[2, :, :] * 0.114
        output[1, :, :] = input[0, :, :] * (-0.168736) + input[1, :, :] * (-0.331264) + input[2, :, :] * 0.5 + 128
        output[2, :, :] = input[ 0, :, :] * 0.5 + input[1, :, :] * (-0.418688) + input[2, :, :] * (-0.081312) + 128
    elif(input.dim()==4):
        output[:,0, :, :] = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114
        output[:,1, :, :] = input[:, 0, :, :] * (-0.168736) + input[:, 1, :, :] * (-0.331264) + input[:, 2, :, :] * 0.5 + 128
        output[:,2, :, :] = input[:, 0, :, :] * 0.5 + input[:, 1, :, :] * (-0.418688) + input[:, 2, :, :] * (-0.081312) + 128
    else:
        print('Wrong image dimension')
        exit()
    output = output/255
    return output
  
def YCbCr_to_RGB(input):
    input = input*255
    output = input.new_empty(input.size())
    output[:, 0, :, :] = input[:, 0, :, :] + (input[:, 2, :, :] - 128) * 1.402
    output[:, 1, :, :] = input[:, 0, :, :] + (input[:, 1, :, :] - 128) * (-0.344136) + (input[:, 2, :, :] - 128) * (-0.714136)
    output[:, 2, :, :] = input[:, 0, :, :] + (input[:, 1, :, :] - 128) * 1.772
    output = output/255
    return output

# ===================Metric=====================
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
    
def PSNR(output,target,max=1.0):
    MAX = torch.tensor(max).to(target.device)
    criterion = nn.MSELoss(reduction='sum').to(target.device)
    MSE = criterion(output,target)/target.numel()
    psnr = 20*torch.log10(MAX)-10*torch.log10(MSE)
    return psnr
    
def BPP(code,num_pixel):    
    return 8*code.numel()/num_pixel
    
# ===================Figure===================== 
def plt_dist(x):
    plt.figure()
    plt.hist(x)
    plt.show()

def plt_result(seed):
    best = load('./output/model/best_{}.pth'.format(seed))
    best_prec1 = best['best_prec1']
    best_epoch = best['best_epoch']
    train_result,test_result,_ = load('./output/result/{}_{}_{}'.format(TAG,seed,best_epoch))
    plt_meter([train_result,test_result],['train','test'],'{}_{}_{}'.format(TAG,seed,best_epoch))
    return
    
def plt_meter(Meters,names,TAG):
    colors = ['r','b']
    print('Figure name: {}'.format(TAG))
    for i in range(len(Meters)):
        fig = plt.figure()
        plt.plot(Meters[i][3].history_avg,label=names[i],color=colors[i])
        plt.legend()
        plt.grid()
        if not os.path.exists('./output/fig/{}'.format(names[i])):
            os.makedirs('./output/fig/{}'.format(names[i]), exist_ok=True) 
        fig.savefig('./output/fig/{}/{}'.format(names[i],TAG), dpi=fig.dpi)

def show(img): 
    npimg = img.cpu().numpy()
    plt.imshow(npimg, interpolation='nearest')
    # plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()