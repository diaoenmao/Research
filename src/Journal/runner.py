import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *



class runner:
    
    regulazition_supported = ['Base','AIC','BIC','BC','GTIC','Lasso','Ridge','ElasticNet','GREG']
    
    def __init__(self, id, data, modelwrappers, ifcuda = True, verbose = True, ifsave = True):

        self.id = id
        self.data = data
        self.modelwrappers = modelwrappers
        self.num_models = len(self.modelwrappers)
        self.ifcuda = ifcuda
        self.verbose = verbose
        self.ifsave = ifsave
        
        self.mode = 'CrossValidation_1'
        self.modename = 'CrossValidation'
        self.input_datatype = torch.FloatTensor
        self.target_datatype = torch.LongTensor
        
    def set_mode(self, mode = 'CrossValidation'):
        self.mode = mode
    
    def set_datatype(self, input_datatype = torch.FloatTensor, target_datatype = torch.LongTensor):
        self.input_datatype = input_datatype
        self.target_datatype = target_datatype

    def set_regularization_param(self, regularization_param):
        self.regularization_param = regularization_param

    def regularize_loss(self,dataSize,model,loss,mode,iftrain):
        if(iftrain):
            if(mode=='Base'):
                regularized_loss = torch.mean(loss)
            elif(mode=='GTIC'):
                regularized_loss = torch.mean(loss) + get_GTIC(dataSize,model,loss)
            elif(mode=='Lasso'):
                regularized_loss = torch.mean(loss) + get_Lasso(model,self.regularization_param[0])
            elif(mode=='Ridge'):
                regularized_loss = torch.mean(loss) + get_Ridge(model,self.regularization_param[1])
            elif(mode=='ElasticNet'):
                regularized_loss = torch.mean(loss) + get_ElasticNet(model,self.regularization_param[:2])
            elif(mode=='GREG'):
                regularized_loss = torch.mean(loss) + get_GREG(dataSize,model,loss,self.regularization_param,np.array([0,1,2]))
            else:
                print('mode not supported while training')
                exit()
        else:
            regularized_loss = get_regularization(dataSize,model,loss,mode,self.regularization_param)
        return regularized_loss
        
    def get_output(self):
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency   
    