import torch
import copy
from torch import nn

class modelWrapper:
    
    def __init__(self,model,optimizer_name='SGD'):
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_param = {'lr':1e-3,'momentum':0,'dampening':0,'weight_decay':0,'nesterov':False,
        'betas':(0.9, 0.999),'eps':1e-8,'amsgrad':False,
        'max_iter':20,'max_eval':None,'tolerance_grad':1e-05,'tolerance_change':1e-09,'history_size':100,'line_search_fn':None}
        self.criterion = nn.NLLLoss(reduce=False)
        
    def set_optimizer_name(self,optimizer_name):
        self.optimizer_name = optimizer_name
        
    def set_optimizer_param(self,optimizer_param):
        self.optimizer_param = {**self.optimizer_param, **optimizer_param}
    
    def set_criterion(self,criterion):
        self.criterion = criterion
        
    def copy(self):
        copied_mw = modelWrapper(copy.deepcopy(self.model),self.optimizer_name)
        copied_mw.set_optimizer_param(self.optimizer_param)
        copied_mw.set_criterion(self.criterion)
        copied_mw.wrap()
        return copied_mw
        
    def wrap(self):
        if(self.optimizer_name=='SGD'):
            self.optimizer = torch.optim.SGD(self.model.parameters(),self.optimizer_param['lr'],self.optimizer_param['momentum'],self.optimizer_param['dampening'],
            self.optimizer_param['weight_decay'],self.optimizer_param['nesterov']) 
        elif(self.optimizer_name=='Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(),self.optimizer_param['lr'],self.optimizer_param['betas'],self.optimizer_param['eps'],
            self.optimizer_param['weight_decay'],self.optimizer_param['amsgrad']) 
        elif(self.optimizer_name=='LBFGS'):
            self.optimizer = torch.optim.LBFGS(self.model.parameters(),self.optimizer_param['lr'],self.optimizer_param['max_iter'],self.optimizer_param['max_eval'],
            self.optimizer_param['tolerance_grad'],self.optimizer_param['tolerance_change'],self.optimizer_param['history_size'],self.optimizer_param['line_search_fn'])
			
def gen_modelwrappers(models,optimizer_param,optimizer_name,criterion):
    modelwrappers = []
    for i in range(len(models)):
        mw = modelWrapper(models[i])
        mw.set_optimizer_name(optimizer_name)
        mw.set_optimizer_param(optimizer_param)
        mw.set_criterion(criterion)
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