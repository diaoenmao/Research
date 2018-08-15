import torch
import torchvision
from torch import nn
from functional import *

activation_mode='elu'

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

class Basic_Conv(nn.Module):
    def __init__(self, inchan, outchan):
        super(Basic_Conv, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        out = self.activation(self.conv(x))
        return out     

class InputTransition(nn.Module):
    def __init__(self, outchan):
        super(InputTransition, self).__init__()
        self.outchan = outchan
        self.conv1 = nn.Conv2d(1, outchan, kernel_size=3, stride=1, padding=1)
        self.activation = Activation(outchan)

    def forward(self, x):
        out = self.conv1(x)
        residual = torch.cat([x for i in range(self.outchan)],1)
        out = self.activation(torch.add(out, residual))
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
        self.downsample = DownSample(inchan, outchan)
        self.conv = _make_Conv(Basic_Conv,outchan,outchan,depth)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        down = self.downsample(x)
        out = self.conv(down)
        out = self.activation(torch.add(out, down))
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
        self.upsample = UpSample(inchan,outchan)
        self.conv = _make_Conv(Basic_Conv,outchan,outchan,depth)
        self.activation = Activation(outchan)

    def forward(self, x):
        up = self.upsample(x)
        out = self.conv(up)
        out = self.activation(torch.add(out, up))
        return out   

        
class OutputTransition(nn.Module):
    def __init__(self, inchan):
        super(OutputTransition, self).__init__()
        self.conv = nn.Conv2d(inchan, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.conv(x))
        return out

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        
    def forward(self,x):
        x = quantizer(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input = InputTransition(32)
        self.down_0 = DownTransition(32,64,1)
        self.down_1 = DownTransition(64,32,1)           
        self.quantizer = Quantizer()

    def forward(self, x_in_0, balance=True):
        x_in_1 = self.input(x_in_0)
        x_in_2 = self.down_0(x_in_1)
        x = self.down_1(x_in_2)
        code = self.quantizer(x)
        if(balance):
            return code,[x_in_0,x_in_1,x_in_2]
        else:
            return code

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_0 = UpTransition(32,64,1) 
        self.up_1 = UpTransition(64,32,1)
        self.output = OutputTransition(32)      

    def forward(self, x, balance=True):
        x_out_2 = self.up_0(x)
        x_out_1 = self.up_1(x_out_2)
        x_out_0 = self.output(x_out_1)
        if(balance):        
            return x_out_0,[x_out_0,x_out_1,x_out_2]
        else:
            return x_out_0
        
class CAE(nn.Module):
    def __init__(self,if_balance=True):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.if_balance = if_balance
        self.balance_loss_fn = nn.MSELoss()
        
    def balance(self, x):
        code,down = self.encoder(x)
        decoded_x,up = self.decoder(code)
        balance_loss = []
        for i in range(len(down)):
            balance_loss.append(self.balance_loss_fn(up[i],down[i]))
        return decoded_x,balance_loss
        
    def forward(self, x,):
        if(self.if_balance):
             decoded_x, balance_loss = self.balance(x)
             return decoded_x,balance_loss
        else:
            code = self.encoder(x,self.if_balance)
            decoded_x = self.decoder(code,self.if_balance)
            return decoded_x
        
def cae(**kwargs):
    model = CAE(**kwargs)
    return model
    