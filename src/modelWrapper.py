import itertools
import torch
import copy
import time
import numpy as np
from torch import nn
from util import *
from modelselect import *

class modelWrapper:
    
    def __init__(self,model,optimizer_name,device):
        self.model = model
        self.optimizer_name = optimizer_name
        self.device = device
        self.optimizer_param = {'lr':1e-3,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,
        'betas':(0.9, 0.999),'eps':1e-8,'amsgrad':False,
        'max_iter':20,'max_eval':None,'tolerance_grad':1e-05,'tolerance_change':1e-09,'history_size':100,'line_search_fn':None}
        self.criterion = nn.NLLLoss(reduce=False)
        self.ifcuda = next(self.model.parameters()).is_cuda
        self.regularization = None
        self.if_optimize_regularization=False
        self.reg_coordinate_set = None
        self.coordinate_set = None
        self.fixed_coordinate = None
        self.active_coordinate = None
        
    def set_optimizer_name(self,optimizer_name):
        self.optimizer_name = optimizer_name
        
    def set_optimizer_param(self,optimizer_param):
        self.optimizer_param = {**self.optimizer_param, **optimizer_param}
    
    def set_criterion(self,criterion):
        self.criterion = criterion
        
    def set_regularization(self,regularization,if_optimize_regularization,regularization_mode='all'):
        self.regularization = regularization
        self.regularization_mode = regularization_mode
        self.if_optimize_regularization = if_optimize_regularization
        self.regularization_mode = regularization_mode
        if(self.regularization is not None): 
            if(self.regularization_mode=='all'):
                self.regularization_parameters = [torch.tensor([regularization[j]],dtype=torch.float,device=self.device,requires_grad=self.if_optimize_regularization) for j in range(len(regularization))]
            elif(self.regularization_mode=='single'):
                self.regularization_parameters = []
                for i in range(len(list(self.model.parameters()))):
                    self.regularization_parameters.extend([torch.tensor([regularization[j]],dtype=torch.float,device=self.device,requires_grad=self.if_optimize_regularization) for j in range(len(regularization))])
        return
        
    def parameters(self):
        if(self.regularization is None):
            return list(self.model.parameters())
        else:
            parameters = [*self.regularization_parameters, *self.model.parameters()]
            return parameters
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_coordinate(self,coordinate_set,fixed_coordinate=None):
        if(self.regularization is not None):
                num_regularization_parameters = len(self.regularization_parameters)
                if(self.if_optimize_regularization): 
                    if(self.regularization_mode=='all'):
                        self.reg_coordinate_set = list(range(len(self.regularization)))
                    elif(self.regularization_mode=='single'):
                        self.reg_coordinate_set = []
                        for i in range(len(coordinate_set)):
                            cur_reg_coordinate_set = []
                            for j in range(len(coordinate_set[i])):  
                                cur_reg_coordinate_set.extend(list(range(coordinate_set[i][j]*len(self.regularization),coordinate_set[i][j]*len(self.regularization)+len(self.regularization)))) 
                            self.reg_coordinate_set.append(cur_reg_coordinate_set)
                else:
                    self.reg_coordinate_set = None
        else:
            num_regularization_parameters = 0
            self.reg_coordinate_set = None
        self.coordinate_set = []
        for i in range(len(coordinate_set)): 
            if(self.regularization_mode=='all'):
                if(self.reg_coordinate_set is not None):
                    self.coordinate_set.append(self.reg_coordinate_set)
            elif(self.regularization_mode=='single'):
                if(self.reg_coordinate_set is not None):
                    self.coordinate_set.append(self.reg_coordinate_set[i])
            self.coordinate_set.append((np.array(coordinate_set[i])+num_regularization_parameters).astype(int).tolist())
        if(fixed_coordinate is not None):
            self.fixed_coordinate = np.array(fixed_coordinate)+num_regularization_parameters
            reg_fixed_coordinate = []
            for i in range(len(fixed_coordinate)):
                reg_fixed_coordinate.extend(list(range(fixed_coordinate[i]*len(self.regularization),fixed_coordinate[i]*len(self.regularization)+len(self.regularization))))
            self.fixed_coordinate = np.hstack((np.array(reg_fixed_coordinate),self.fixed_coordinate)).astype(int).tolist()
        else:
            self.fixed_coordinate = []
        return  
                  
    def activate_coordinate(self,coordinate):
        param = self.parameters()
        if(self.active_coordinate is None):
            all_coordinate = list(range(len(param)))
            for i in all_coordinate:
                param[i].requires_grad_(False)
        else:
            for i in self.active_coordinate:
                param[i].requires_grad_(False)
        self.active_coordinate = coordinate
        for i in self.active_coordinate:
            param[i].requires_grad_(True)                    
        return
        
    def GTIC(self,loss_batch,coordinate):
        #print(coordinate)
        dataSize = loss_batch.size(0)
        likelihood_batch = -loss_batch
        param = self.parameters()
        if(coordinate is None):
            local_param = param
        else:
            local_param = [param[i] for i in coordinate]
        #print(local_param)
        list_vec_free_grad_params=[]
        for j in range(dataSize):
            grad_params = torch.autograd.grad(likelihood_batch[j], local_param, create_graph = True)
            vec_grad_params = torch.cat(grad_params,dim=0)
            vec_grad_params = vec_grad_params.unsqueeze(1).unsqueeze(0)
            list_vec_free_grad_params.append(vec_grad_params)         
        grad_params = torch.cat(list_vec_free_grad_params,dim=0)
        sum_grad_params = torch.sum(grad_params,dim=0)
        non_zero_idx_J = torch.nonzero(sum_grad_params[:,0])
        if(non_zero_idx_J.size()==torch.Size([])):
            print('empty J')
            return 0
        grad_params_T = grad_params.transpose(1,2)
        J_batch = torch.matmul(grad_params,grad_params_T)
        J = torch.sum(J_batch,dim=0)
        J = J[non_zero_idx_J,non_zero_idx_J.view(1,-1)]   
        J = J/dataSize
        H = []
        for j in sum_grad_params:
            h = torch.autograd.grad(j, local_param, create_graph=True)
            vec_h = torch.cat(h,dim=0)
            vec_h = vec_h.unsqueeze(0)
            H.append(vec_h)
        H = torch.cat(H,dim=0)
        H = H[non_zero_idx_J,non_zero_idx_J.view(1,-1)]
        sum_H = torch.sum(H,dim=0)
        non_zero_idx_H = torch.nonzero(sum_H)
        if(non_zero_idx_H.size()==torch.Size([])):
            print('empty H')
            return 0
        J = J[non_zero_idx_H,non_zero_idx_H.view(1,-1)] 
        H = H[non_zero_idx_H,non_zero_idx_H.view(1,-1)]
        # print('J')
        # print(J)
        # print('H')
        # print(H)
        V = -H/dataSize
        try:
            inv_V = torch.inverse(V)
            VmJ = torch.matmul(inv_V,J)
            tVMJ = torch.trace(VmJ)
            GTIC = tVMJ/dataSize
            if(GTIC < 0):
                print('numerically unstable, negative')
                #print('effective num of paramters')
                #print(float(tVMJ))
                GTIC = 0
            else:
                a = 0
               # print('effective num of paramters')
                #print(float(tVMJ))
        except RuntimeError as e:
            print(e)
            print('numerically unstable, not invertable')
            GTIC = 0
        #exit()
        return GTIC
        
    def L(self,input,target,if_eval,if_GTIC=False):        
        model = self.model.eval() if if_eval else self.model
        output = self.model(input)
        loss_batch = self.criterion(output, target)
        loss = torch.mean(loss_batch)
        REG = 0
        GTIC = 0
        if(self.regularization is not None):
            i = 0
            for p in self.model.parameters():
                if(self.regularization_mode=='all'):
                    for j in range(len(self.regularization)):
                        REG = REG + torch.exp(self.regularization_parameters[j]) * p.norm(np.float(j+1))
                elif(self.regularization_mode=='single'):
                    for j in range(len(self.regularization)):
                        REG = REG + torch.exp(self.regularization_parameters[i]) * p.norm(np.float(j+1))  
                        i = i + 1
        if(if_GTIC):
            free_coordinate = list(set(self.active_coordinate)-set(self.fixed_coordinate))
            GTIC = self.GTIC(loss_batch+REG,free_coordinate)
        regularized_loss = loss + REG + GTIC
        return loss,regularized_loss
        
    def acc(self,input,target):
        output = self.model(input)
        acc = get_acc(output,target)
        return acc
        
    def wrap(self):
        if(self.coordinate_set is None or len(self.coordinate_set)==0):
            if(self.optimizer_name=='SGD'):
                self.optimizer = torch.optim.SGD(self.parameters(),self.optimizer_param['lr'],self.optimizer_param['momentum'],self.optimizer_param['dampening'],
                self.optimizer_param['weight_decay'],self.optimizer_param['nesterov']) 
            elif(self.optimizer_name=='Adam'):
                self.optimizer = torch.optim.Adam(self.parameters(),self.optimizer_param['lr'],self.optimizer_param['betas'],self.optimizer_param['eps'],
                self.optimizer_param['weight_decay'],self.optimizer_param['amsgrad']) 
        else:          
            self.optimizer=[]
            param = list(self.parameters())
            for i in range(len(self.coordinate_set)):               
                cur_param = [param[j] for j in self.coordinate_set[i]]
                if(self.optimizer_name=='SGD'):
                    self.optimizer.append(torch.optim.SGD(cur_param,self.optimizer_param['lr'],self.optimizer_param['momentum'],self.optimizer_param['dampening'],
                    self.optimizer_param['weight_decay'],self.optimizer_param['nesterov']))
                elif(self.optimizer_name=='Adam'):
                    self.optimizer.append(torch.optim.Adam(cur_param,self.optimizer_param['lr'],self.optimizer_param['betas'],self.optimizer_param['eps'],
                    self.optimizer_param['weight_decay'],self.optimizer_param['amsgrad']))
                else:
                    print('Optimizer not supported')
                    exit()
        return
    
    
