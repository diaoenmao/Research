import math
import torch
import copy
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
    
 
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
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
            mesh_indices = tuple(np.meshgrid(*valid_indices, sparse=False, indexing='ij'))
            raveled_indices = np.ravel_multi_index(mesh_indices, dims=(self.out_features,self.in_features+1), order='C')
            raveled_indices = raveled_indices.ravel()
            coordinate_set.append(raveled_indices)
            pivot = pivot+local_size
        return coordinate_set
    
    def fixed_coordinate(self):
        if(self.out_features==1):
            return None
        input_indices = np.arange(self.in_features+1)
        out_index = self.out_features-1
        fixed_indices = [out_index,input_indices]
        mesh_indices = tuple(np.meshgrid(*fixed_indices, sparse=False, indexing='ij'))
        raveled_indices = np.ravel_multi_index(mesh_indices, dims=(self.out_features,self.in_features+1), order='C') 
        fixed_coordinate = raveled_indices.ravel().tolist()
        return fixed_coordinate
        
    def forward(self, input):
        self.matrix = torch.cat(list(self.parameters()), dim=0).view(self.out_features,self.in_features+1)
        self.weight = self.matrix[:,:self.in_features]
        self.bias = self.matrix[:,-1]
        return nn.functional.linear(input, self.weight, self.bias)
        