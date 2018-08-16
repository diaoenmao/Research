import itertools
import torch
import copy
import time
import numpy as np
from torch import nn
from util import *

class modelWrapper:
    
    def __init__(self,model,optimizer_name):
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_param = {'lr':1e-1,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,
        'betas':(0.9, 0.999),'eps':1e-8,'amsgrad':False,
        'max_iter':20,'max_eval':None,'tolerance_grad':1e-05,'tolerance_change':1e-09,'history_size':100,'line_search_fn':None}
        self.regularization = None
        self.active_coordinate = None
        self.norm1_lagrange = 1e-4
        
    def set_optimizer_name(self,optimizer_name):
        self.optimizer_name = optimizer_name
        
    def set_optimizer_param(self,optimizer_param):
        self.optimizer_param = {**self.optimizer_param, **optimizer_param}
                
    def set_criterion(self,criterion):
        self.criterion = criterion

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
        
    def parameters(self):
        return list(self.model.parameters())
        
    def num_free_parameters(self):
        last_layer_weight = self.parameters()[-2]
        return self.num_parameters() - (last_layer_weight.size(1) + 1)
    
    def free_parameters(self,parameters=None):
        if(parameters is None):
            free_parameters = self.parameters()
        else:
            free_parameters = list(parameters)
        free_parameters[-2] = free_parameters[-2][:-1,]
        free_parameters[-1] = free_parameters[-1][:-1]
        return free_parameters
    
    def loss(self,output,target):
        #loss = self.criterion(input,output[1]) + self.norm1_lagrange*output[0].norm(p=1)
        loss = self.criterion(output,target)
        return loss
        
    def acc(self,output,target,topk=(1,)):
        acc = get_acc(output,target,topk=topk)
        return acc
    
    def set_optimizer(self):
        param = self.model.parameters()
        optimizer_dict = self.optimizer_param
        if(self.optimizer_name=='SGD'):
            self.optimizer = torch.optim.SGD(param,optimizer_dict['lr'],optimizer_dict['momentum'],optimizer_dict['dampening'],
            optimizer_dict['weight_decay'],optimizer_dict['nesterov'])
        elif(self.optimizer_name=='Adam'):
            self.optimizer = torch.optim.Adam(param,optimizer_dict['lr'],optimizer_dict['betas'],optimizer_dict['eps'],
            optimizer_dict['weight_decay'],optimizer_dict['amsgrad'])
        elif(self.optimizer_name=='LBFGS'):
            self.optimizer = torch.optim.LBFGS(param,optimizer_dict['lr'],optimizer_dict['max_iter'],optimizer_dict['max_eval'],
            optimizer_dict['tolerance_grad'],optimizer_dict['tolerance_change'],optimizer_dict['history_size'],optimizer_dict['line_search_fn'])
        else:
            print('Optimizer not supported')
            exit()
        return
    
    
