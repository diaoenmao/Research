import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class Linear(nn.Module):
    def __init__(self,input_feature,output_feature):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(input_feature, output_feature)

    def forward(self, x):
        x = self.fc1(x)
        return x
        
def linear(**kwargs):
    model = Linear(**kwargs)
    return model