import torch
import numpy as np
from torch import nn

def passthrough(x, **kwargs):
    return x
    
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, if_maxpool=True):
        super(conv, self).__init__()
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        if(if_maxpool):
            self.deep = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.deep = passthrough
                
    def forward(self, x):
        x = self.kernel(x)
        x = self.deep(x)
        return x

class linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(linear, self).__init__()
        self.kernel = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
                
    def forward(self, x):
        x = self.kernel(x)
        return x
    
class Organic_Conv11(nn.Module):

    def __init__(self,  num_classes=1000, init_weights=True):
            super(Organic_Conv11, self).__init__()
            self.features = nn.Sequential(
                conv(3,64,True),
                
                Organic(64),
                #nn.Dropout(),                 
                conv(64,128,True),
                
                #Organic(128),                
                conv(128,256,False),                
                #Organic(256),
                conv(256,256,True), 
                
                #Organic(256),              
                conv(256,512,False),
                #Organic(512),              
                conv(512,512,True),
                
                #Organic(512),               
                conv(512,512,False),
                #Organic(512),                
                conv(512,512,True)
            )
            self.classifier = nn.Sequential(
                linear(512,4096),
                #Organic(4096),
                nn.Dropout(),
                linear(4096,4096),
                #Organic(4096),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        
def organic_conv11(**kwargs):
    model = Organic_Conv11(**kwargs)
    return model
    
    
class Dropout_Conv11(nn.Module):

    def __init__(self,  num_classes=1000, init_weights=True):
            super(Dropout_Conv11, self).__init__()
            self.features = nn.Sequential(
                conv(3,64,True),
                
                nn.Dropout(),                
                conv(64,128,True),
                
                nn.Dropout(),                
                conv(128,256,False),                
                nn.Dropout(),
                conv(256,256,True), 
                 
                nn.Dropout(),                
                conv(256,512,False),  
                nn.Dropout(),                
                conv(512,512,True),
                
                nn.Dropout(),                
                conv(512,512,False),
                nn.Dropout(),                
                conv(512,512,True)
            )
            self.classifier = nn.Sequential(
                linear(512,4096),
                nn.Dropout(),
                linear(4096,4096),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        
def dropout_conv11(**kwargs):
    model = Dropout_Conv11(**kwargs)
    return model