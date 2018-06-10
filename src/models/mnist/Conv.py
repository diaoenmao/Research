import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from Organic import *

def passthrough(x, **kwargs):
    return x

class Dropout_Conv(nn.Module):
    def __init__(sel,num_classes=10):
        super(Dropout_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x

        
def dropout_conv(**kwargs):
    model = Dropout_Conv(**kwargs)
    return model
    
class Organic_Conv(nn.Module):
    def __init__(self,num_classes=10):
        super(Organic_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_organic = Organic(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc1_organic = Organic(50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_organic(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_organic(x)
        x = self.fc2(x)
        return x

        
def organic_conv(**kwargs):
    model = Organic_Conv(**kwargs)
    return model