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
        self.optimizer_param = {'lr':1e-3,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,
        'betas':(0.9, 0.999),'eps':1e-8,'amsgrad':False,
        'max_iter':20,'max_eval':None,'tolerance_grad':1e-05,'tolerance_change':1e-09,'history_size':100,'line_search_fn':None}
        self.criterion = nn.NLLLoss(reduce=False)
        self.regularization = None
        self.active_coordinate = None
        
    def set_optimizer_name(self,optimizer_name):
        self.optimizer_name = optimizer_name
        
    def set_optimizer_param(self,optimizer_param):
        self.optimizer_param = {**self.optimizer_param, **optimizer_param}
                
    def set_criterion(self,criterion):
        self.criterion = criterion
        
    def parameters(self):
        return list(self.model.parameters())
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
        
    def loss(self,output,target):        
        loss = self.criterion(output, target)
        return loss
        
    def acc(self,output,target,topk=(1,)):
        acc = get_acc(output,target,topk=topk)
        return acc
    
    def gen_optimizer(self,optimizer_name,param,optimizer_dict):
        if(optimizer_name=='SGD'):
            opt = torch.optim.SGD(param,optimizer_dict['lr'],optimizer_dict['momentum'],optimizer_dict['dampening'],
            optimizer_dict['weight_decay'],optimizer_dict['nesterov'])
        elif(self.optimizer_name=='Adam'):
            opt = torch.optim.Adam(param,optimizer_dict['lr'],optimizer_dict['betas'],optimizer_dict['eps'],
            optimizer_dict['weight_decay'],optimizer_dict['amsgrad'])
        elif(self.optimizer_name=='LBFGS'):
            opt = torch.optim.LBFGS(param,optimizer_dict['lr'],optimizer_dict['max_iter'],optimizer_dict['max_eval'],
            optimizer_dict['tolerance_grad'],optimizer_dict['tolerance_change'],optimizer_dict['history_size'],optimizer_dict['line_search_fn'])
        else:
            print('Optimizer not supported')
            exit()
        return opt
        
    def wrap(self):
        self.optimizer = self.gen_optimizer(self.optimizer_name,self.model.parameters(),self.optimizer_param)
        return
    
    
