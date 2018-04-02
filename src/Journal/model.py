import torch
import copy
from torch import nn
from modelWrapper import *

class Linear(nn.Module):
    def __init__(self,in_features, out_features, ifclassification):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.ifclassification = True
        self.outputlayer = self.linear
		
    def forward(self, x):
        x = self.linear(x)
        return x
        
def gen_models_Linear(input_features,out_features,input_datatype,ifclassification,ifcuda):
    models = []
    for i in range(len(input_features)):
        model = Linear(input_features[i].shape[0],out_features[i],ifclassification).type(input_datatype).cuda() if ifcuda else Linear(input_features[i].shape[0],out_features[i],ifclassification).type(input_datatype)
        models.append(model)
    return models
