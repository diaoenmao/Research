import torch
import copy
from torch import nn

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
    
    