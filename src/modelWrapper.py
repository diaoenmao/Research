import itertools
import torch
import copy
from torch import nn
from util import *
from modelselect import *

class modelWrapper:
    
    def __init__(self,model,optimizer_name):
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_param = {'lr':1e-3,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,
        'betas':(0.9, 0.999),'eps':1e-8,'amsgrad':False,
        'max_iter':20,'max_eval':None,'tolerance_grad':1e-05,'tolerance_change':1e-09,'history_size':100,'line_search_fn':None}
        self.criterion = nn.NLLLoss(reduce=False)
        self.regularization = None
        self.regularization_parameters = None
        self.if_joint_regularization = False
        self.ifcuda = next(self.model.parameters()).is_cuda
        
    def set_optimizer_name(self,optimizer_name):
        self.optimizer_name = optimizer_name
        
    def set_optimizer_param(self,optimizer_param):
        self.optimizer_param = {**self.optimizer_param, **optimizer_param}
    
    def set_criterion(self,criterion):
        self.criterion = criterion
        
    def set_regularization(self,regularization_parameters,if_joint_regularization):
        self.regularization = regularization_parameters
        self.if_joint_regularization = if_joint_regularization
        if(self.regularization is not None and len(self.regularization)>1): 
            self.regularization_parameters = to_var(torch.FloatTensor(regularization_parameters[1:]),self.ifcuda,self.if_joint_regularization)
            
    def parameters(self):
        if(self.regularization is None or self.regularization_parameters is None or not self.if_joint_regularization):
            return list(self.model.parameters())
        else:
            parameters = [self.regularization_parameters, *self.model.parameters()]
            return parameters
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def free_parameters(self, param):
        if(not self.model.ifclassification):
            return param
        num_outputlayer_param = len(list(self.model.outputlayer.parameters()))
        free_param = param[:-num_outputlayer_param]
        outputlayer_param = param[-num_outputlayer_param:]
        outputlayer_free_param = []
        for p in outputlayer_param:
            if(p.size()[0]>1):
                outputlayer_free_param.append(p[:-1,])
            else:
                outputlayer_free_param.append(p)
        free_param.extend(outputlayer_free_param)
        return free_param
    
    def num_free_parameters(self, param):
        free_param = self.free_parameters(param)
        vec_free_params = vectorize_parameters(free_param)
        num_free_params = vec_free_params.size()[0]
        return num_free_params

    def free_vec_parameters_idx(self):
        param = self.parameters()
        outputlayer_param = list(self.model.outputlayer.parameters())
        num_outputlayer_param = len(outputlayer_param)
        free_param = param[:-num_outputlayer_param]
        count = 0
        idx = None
        if(len(free_param)!=0):
            for i in range(len(free_param)):
                count = count + torch.numel(free_param[i])
            idx = torch.arange(count).long()
        for p in outputlayer_param:
            if(p.size()[0]>1):
                cur_num_free_param = torch.numel(p[:-1,])
            else:
                cur_num_free_param = torch.numel(p)
            total_num_free_param = torch.numel(p)
            if(idx is None):
                idx = torch.arange(count,count+cur_num_free_param).long()
            else:
                idx = torch.cat((idx,torch.arange(count,count+cur_num_free_param).long()),dim=0)
            count = count+total_num_free_param
        if(self.ifcuda):
            idx = idx.cuda()
        return idx
    
    def loss_acc(self,input,target):
        dataSize = input.size()[0]
        output = self.model(input)
        loss_batch = self.criterion(output, target)
        loss = torch.mean(loss_batch)
        acc = get_acc(output,target)
        if(self.regularization is not None):
            regularized_loss = loss + get_REG(dataSize,self,loss_batch,self.regularization,self.regularization_parameters)     
        else:
            regularized_loss = loss
        return loss,regularized_loss,loss_batch,acc
        
    def copy(self):
        copied_mw = modelWrapper(copy.deepcopy(self.model),self.optimizer_name)
        copied_mw.set_optimizer_param(self.optimizer_param)
        copied_mw.set_criterion(self.criterion)
        copied_mw.set_regularization(self.regularization,self.if_joint_regularization)
        copied_mw.wrap()
        return copied_mw
        
    def wrap(self):
        if(self.optimizer_name=='SGD'):
            self.optimizer = torch.optim.SGD(self.parameters(),self.optimizer_param['lr'],self.optimizer_param['momentum'],self.optimizer_param['dampening'],
            self.optimizer_param['weight_decay'],self.optimizer_param['nesterov']) 
        elif(self.optimizer_name=='Adam'):
            self.optimizer = torch.optim.Adam(self.parameters(),self.optimizer_param['lr'],self.optimizer_param['betas'],self.optimizer_param['eps'],
            self.optimizer_param['weight_decay'],self.optimizer_param['amsgrad']) 
        elif(self.optimizer_name=='LBFGS'):
            self.optimizer = torch.optim.LBFGS(self.parameters(),self.optimizer_param['lr'],self.optimizer_param['max_iter'],self.optimizer_param['max_eval'],
            self.optimizer_param['tolerance_grad'],self.optimizer_param['tolerance_change'],self.optimizer_param['history_size'],self.optimizer_param['line_search_fn'])
			
def gen_modelwrappers(models,optimizer_param,optimizer_name,criterion,regularization_parameters=None,if_joint_regularization=False):
    modelwrappers = []
    for i in range(len(models)):
        mw = modelWrapper(models[i],optimizer_name)
        mw.set_optimizer_param(optimizer_param)
        mw.set_criterion(criterion)
        mw.set_regularization(regularization_parameters,if_joint_regularization)
        mw.wrap()
        modelwrappers.append(mw)
    return modelwrappers
                
def unpack_modelwrappers(modelwrappers):
    models = []
    optimizer = []
    for i in range(len(modelwrappers)):
        models.append(modelwrappers[i].model)
        optimizer.append(modelwrappers[i].optimizer)
    return models,optimizer
    
    
