import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *



class runner:
    
    regulazition_supported = ['Base','AIC','BIC','BC','GTIC','REG']
    
    def __init__(self, id, data, modelwrappers, ifcuda = True, verbose = True, ifsave = True):

        self.id = id
        self.data = data
        self.modelwrappers = modelwrappers
        self.num_models = len(self.modelwrappers)
        self.ifcuda = ifcuda
        self.verbose = verbose
        self.ifsave = ifsave
        
        self.mode = 'CrossValidation'
        self.input_datatype = torch.FloatTensor
        self.target_datatype = torch.LongTensor
        
    def set_mode(self, mode = 'CrossValidation'):
        self.mode = mode
    
    def set_datatype(self, input_datatype = torch.FloatTensor, target_datatype = torch.LongTensor):
        self.input_datatype = input_datatype
        self.target_datatype = target_datatype

    def set_ifregularize(self, ifregularize):
        self.ifregularize = ifregularize
        
    def get_output(self):
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency   
    