import torch
import torchvision
from torch import nn
from functional import *

activation_mode='relu'

def _make_Conv(conv, inchan, outchan, depth):
    layers = []
    for i in range(depth):
        if(i==0):
            layers.append(conv(inchan, outchan))
        else:
            layers.append(conv(outchan, outchan))
    return nn.Sequential(*layers)
    
def Activation(nchan):
    if(activation_mode=='relu'):
        return nn.ReLU(True)
    elif(activation_mode=='elu'):
        return nn.ELU(True)
    elif(activation_mode=='prelu'):
        return nn.PReLU(nchan)
    else:
        print('Not suppported activation mode')
        exit()

class FC_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(FC_Conv, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, padding=0)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        out = self.activation(self.conv(x))
        return out

class Basic_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(Basic_Conv, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        out = self.activation(self.conv(x))
        return out
        
class DownSample(nn.Module):
    def __init__(self, inchan, outchan):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=2, stride=2, padding=0) 
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x = self.activation(self.conv(x))
        return x

class DownTransition(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(DownTransition, self).__init__()
        self.convs = _make_Conv(Basic_Conv,inchan,outchan,depth)
        self.fc = FC_Conv(inchan,outchan)
        self.downsample = DownSample(outchan,outchan)

    def forward(self, x):
        out = self.convs(x)
        residual = self.fc(x)
        out = out + residual
        out = self.downsample(out)
        return out
                
class UpSample(nn.Module):
    def __init__(self, inchan, outchan):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(inchan, outchan, kernel_size=2, stride=2, padding=0)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
                
class UpTransition(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(UpTransition, self).__init__()
        self.convs = _make_Conv(Basic_Conv,inchan,outchan,depth)
        self.fc = FC_Conv(inchan,outchan)
        self.upsample = UpSample(inchan,inchan)

    def forward(self, x):
        x = self.upsample(x)
        residual = self.fc(x)
        out = self.convs(x)
        out = out + residual
        return out        

class OutputTransition(nn.Module):
    def __init__(self, inchan, outchan):
        super(OutputTransition, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1)
        self.fc = FC_Conv(inchan,outchan)
        self.upsample = UpSample(inchan,inchan)

    def forward(self, x):
        x = self.upsample(x)
        residual = self.fc(x)
        out = self.conv(x)
        out = out + residual
        return out
    
class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        
    def forward(self,x):
        x = quantizer(x)
        return x

# class CAE(nn.Module):
    # def __init__(self):
        # super(CAE, self).__init__()
        # self.encoder = nn.Sequential(
            # nn.Conv2d(1, 32, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(32, 32, 2, stride=2, padding=0),
            # nn.ReLU(True),
            # nn.Conv2d(32, 64, 3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 2, stride=2, padding=0),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(64, 32, 2, stride=2, padding=0),               
        # )
        # self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 64, 2, stride=2, padding=0),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
            # nn.ReLU(True),
            # nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0), 
            # nn.ReLU(True),
            # nn.Conv2d(32, 1, 3, stride=1, padding=1), 
            # nn.Tanh()
        # )
        # self.quantizer = Quantizer()

    # def forward(self, x):
        # x = self.encoder(x)
        # x = self.quantizer(x)
        # x = self.decoder(x)
        # return x

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            DownTransition(1,32,1),
            DownTransition(32,64,1), 
            DownTransition(64,32,1),            
        )
        self.decoder = nn.Sequential(
            UpTransition(32,64,1),
            UpTransition(64,32,1),
            OutputTransition(32,1),
            nn.Tanh()
        )
        self.quantizer = Quantizer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.decoder(x)
        return x
        
def cae(**kwargs):
    model = CAE(**kwargs)
    return model
    