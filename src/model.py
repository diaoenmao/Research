import math
import torch
import copy
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    def __init__(self, in_features, out_features, ifclassification):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ifclassification = ifclassification
        
        self.linear = nn.Linear(self.in_features, self.out_features)
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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_layers, out_features, ifclassification):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.ifclassification = ifclassification        
        
        self.linears = nn.ModuleList([nn.Linear(self.in_features, self.hidden_layers[0]),nn.ReLU()])
        for i in range(len(self.hidden_layers)-1):
            self.linears.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.linears.append(nn.ReLU())
        self.outputlayer = nn.Linear(self.hidden_layers[-1], self.out_features)
        self.linears.append(self.outputlayer)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
        return x
        
def gen_models_MLP(input_features,hidden_layers,out_features,input_datatype,ifclassification,ifcuda):
    models = []
    for i in range(len(hidden_layers)):
        model = MLP(input_features[i],hidden_layers[i],out_features[i],ifclassification).type(input_datatype).cuda() if ifcuda else MLP(input_features[i],hidden_layers[i],out_features[i],ifclassification).type(input_datatype)
        models.append(model)
    return models
    
 
class local_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(local_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parameters = (self.in_features+1)*(self.out_features)
        for i in range(self.num_parameters):
            self.register_parameter(str(i), Parameter(torch.Tensor(1,))) 
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def coordinate_set(self,local_size):
        coordinate_set = []
        out_indices = np.arange(self.out_features)
        pivot = list(range(0,self.in_features+1,local_size))
        pivot = 0
        coordinate_set = []
        while(pivot<=self.in_features):
            input_indices = np.arange(pivot,pivot+local_size)
            valid_input_indices = input_indices[(input_indices>=0)&(input_indices<=self.in_features)]
            valid_indices = [out_indices,valid_input_indices]
            mesh_indices = np.meshgrid(*valid_indices, sparse=False, indexing='ij')
            mesh_indices = tuple(mesh_indices)
            raveled_indices = np.ravel_multi_index(mesh_indices, dims=(self.out_features,self.in_features+1), order='C') 
            raveled_indices = raveled_indices.ravel()
            coordinate_set.append(raveled_indices)
            pivot = pivot+local_size
        return coordinate_set
        
    def forward(self, input):
        self.matrix = torch.cat(list(self.parameters()), dim=0).view(self.out_features,self.in_features+1)
        self.weight = self.matrix[:,:self.in_features]
        self.bias = self.matrix[:,-1]
        return nn.functional.linear(input, self.weight, self.bias)
        