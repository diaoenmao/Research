import torch
import torchvision
from torch import nn
from functional import *

activation_mode='prelu'

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
        return nn.RELU(inplace=True)
    elif(activation_mode=='elu'):
        return nn.ELU(inplace=True)
    elif(activation_mode=='prelu'):
        return nn.PReLU(nchan)
    else:
        print('Not suppported activation mode')
        exit()

class FC_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(FC_Conv, self).__init__()
        self.bn = nn.BatchNorm2d(inchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x_aug = self.dp(self.bn(x))
        out = self.activation(self.conv(x_aug))
        return out

class Basic_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(Basic_Conv, self).__init__()
        self.bn = nn.BatchNorm2d(inchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x_aug = self.dp(self.bn(x))
        out = self.activation(self.conv(x_aug))
        return out
        
class DownSample(nn.Module):
    def __init__(self, inchan, outchan):
        super(DownSample, self).__init__()
        self.bn = nn.BatchNorm2d(inchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=2, stride=2, padding=0, bias=False)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x_aug = self.dp(self.bn(x))
        x = self.activation(self.conv(x_aug))
        return x

class DownTransition(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(DownTransition, self).__init__()
        self.convs = _make_Conv(Basic_Conv,inchan,outchan,depth)
        self.fc = FC_Conv(inchan,outchan)
        self.downsample = DownSample(outchan,outchan)

    def forward(self, x):
        out = self.convs(x)
        out += self.fc(x)
        out = self.downsample(out)
        return out
                
class UpSample(nn.Module):
    def __init__(self, inchan, outchan):
        super(UpSample, self).__init__()
        self.bn = nn.BatchNorm2d(inchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.ConvTranspose2d(inchan, outchan, kernel_size=2, stride=2, padding=0, bias=False)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x_aug = self.dp(self.bn(x))
        x = self.activation(self.conv(x_aug))
        return x
                
class UpTransition(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(UpTransition, self).__init__()
        self.convs = _make_Conv(Basic_Conv,inchan,outchan,depth)
        self.fc = FC_Conv(inchan,outchan)
        self.upsample = UpSample(outchan,outchan)

    def forward(self, x):
        out = self.convs(x)
        out += self.fc(x)
        out = self.upsample(out)
        return out        

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        
    def forward(self,x):
        x = quantizer(x)
        return x
    
    
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            DownTransition(3, 16, 1),
            DownTransition(16, 32, 2),
            DownTransition(32, 64, 2),
            DownTransition(64, 128, 3)
        )
        self.decoder = nn.Sequential(
            UpTransition(128, 64, 3),
            UpTransition(64, 32, 2),
            UpTransition(32, 16, 2),
            UpTransition(16, 3, 1)
        )
        self.quantizer = Quantizer()
        
    def code(self, x):
        encoded_x = self.encoder(x)
        code = self.quantizer(encoded_x)
        return code

    def decode(self, code):
        decode_x = self.decoder(code)
        return decoded_x
        
    def forward(self, x):
        encoded_x = self.encoder(x)
        code = self.quantizer(encoded_x)
        decoded_x = self.decoder(code)
        return encoded_x,decoded_x

def cae(**kwargs):
    model = CAE(**kwargs)
    return model
    