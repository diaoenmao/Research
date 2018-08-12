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
        return nn.ReLU()
    elif(activation_mode=='elu'):
        return nn.ELU()
    elif(activation_mode=='prelu'):
        return nn.PReLU(nchan)
    else:
        print('Not suppported activation mode')
        exit()

class FC_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(FC_Conv, self).__init__()
        self.bn = nn.BatchNorm2d(outchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        out = self.activation(self.bn(self.conv(self.dp(x))))
        return out

class Basic_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(Basic_Conv, self).__init__()
        self.bn = nn.BatchNorm2d(outchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        out = self.activation(self.bn(self.conv(self.dp(x))))
        return out
        
class DownSample(nn.Module):
    def __init__(self, inchan, outchan):
        super(DownSample, self).__init__()
        self.bn = nn.BatchNorm2d(outchan)
        self.dp = nn.Dropout2d()
        self.mp = nn.MaxPool2d(2, stride=2)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x = self.activation(self.bn(self.mp(self.dp(x))))
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
        self.bn = nn.BatchNorm2d(outchan)
        self.dp = nn.Dropout2d()
        self.conv = nn.ConvTranspose2d(inchan, outchan, kernel_size=2, stride=2, padding=0)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x = self.activation(self.bn(self.conv(self.dp(x))))
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
    
    
# class CAE(nn.Module):
    # def __init__(self, init_weights=True):
        # super(CAE, self).__init__()
        # self.encoder = nn.Sequential(
            # DownTransition(3, 32, 1),
            # DownTransition(32, 64, 1),
            # DownTransition(64, 32, 1)
        # )
        # self.decoder = nn.Sequential(
            # UpTransition(32, 64, 1),
            # UpTransition(64, 32, 1),
            # UpTransition(32, 3, 1)
        # )
        # self.quantizer = Quantizer()
        # if init_weights:
                # self._initialize_weights()
        
    # def code(self, x):
        # x = self.encoder(x)
        # x = self.quantizer(x)
        # return x

    # def decode(self, x):
        # x = self.decoder(x)
        # return x
        
    # def _initialize_weights(self):
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)
        # return
        
    # def forward(self, x):
        # x = self.encoder(x)
        # x = self.quantizer(x)
        # x = self.decoder(x)
        # return x

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),       
            nn.ReLU(True),
            nn.Conv2d(64, 64, 2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 2, stride=2, padding=0),           
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0), 
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1), 
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
    