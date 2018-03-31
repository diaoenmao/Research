import torch
import copy
from torch import nn
from modelWrapper import *

class LogisticRegression(nn.Module):
    def __init__(self,in_features, out_features):
        super(LogisticRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.ifclassification = True
        self.outputlayer = self.linear
    def forward(self, x):
        x = self.linear(x)
        return x
        
def gen_models_LogisticRegression(input_features,out_features,input_datatype,ifcuda):
    models = []
    for i in range(len(input_features)):
        model = LogisticRegression(input_features[i].shape[0],out_features[i]).type(input_datatype).cuda() if ifcuda else LogisticRegression(input_features[i].shape[0],out_features[i]).type(input_datatype)
        models.append(model)
    return models
    
def gen_modelwrappers(models,optimizer_param,optimizer_name='SGD'):
    modelwrappers = []
    for i in range(len(models)):
        mw = modelWrapper(models[i])
        mw.set_optimizer_name(optimizer_name)
        mw.set_optimizer_param(optimizer_param)
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