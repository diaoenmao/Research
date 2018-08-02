import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def passthrough(x, **kwargs):
    return x

class _conv(nn.Module):
    def __init__(self, in_channels, out_channels, if_maxpool=True):
        super(_conv, self).__init__()
        self.kernel = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
        if(if_maxpool):
            self.deep = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.deep = passthrough
                
    def forward(self, x):
        x = self.kernel(x)
        x = self.deep(x)
        return x
        
class Conv(nn.Module):

    def __init__(self, input_feature, output_feature, hidden_feature, init_weights=True):
            super(Conv, self).__init__()
            self.bn = nn.BatchNorm1d(input_feature)
            self.features = nn.Sequential(
                _conv(input_feature,hidden_feature,False),
                _conv(hidden_feature,hidden_feature*2,True),  
                
                _conv(hidden_feature*2,hidden_feature*2,False),
                _conv(hidden_feature*2,hidden_feature*4,True), 
                
                _conv(hidden_feature*4,hidden_feature*4,False),                
                _conv(hidden_feature*4,hidden_feature*8,True), 
                            
                _conv(hidden_feature*8,hidden_feature*8,False),             
                _conv(hidden_feature*8,hidden_feature*8,True),
                            
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_feature*8, output_feature)
            )
            if init_weights:
                self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.bn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        
def conv(**kwargs):
    model = Conv(**kwargs)
    return model    
    
    