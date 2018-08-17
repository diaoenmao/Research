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
        self.conv = nn.Conv2d(inchan, outchan*4, kernel_size=3, stride=1, padding=1)
        self.subpix = nn.PixelShuffle(2)
        self.activation = Activation(outchan)
        
    def forward(self, x):
        x = self.activation(self.subpix(self.conv(x)))
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
        self.down_1 = DownTransition(64,64,2)
        self.down_2 = DownTransition(64,32,3)        
        self.quantizer = Quantizer()

    def forward(self, x):
        x = self.input(x)
        x = self.down_0(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.quantizer(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_0 = UpTransition(32,64,3) 
        self.up_1 = UpTransition(64,64,2)
        self.up_2 = UpTransition(64,32,1)
        self.output = OutputTransition(32)      

    def forward(self, x):
        x = self.up_0(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.output(x)
        return x
        
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantizer = Quantizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.decoder(x)
        return x

# class CAE(nn.Module):
    # def __init__(self):
        # super(CAE, self).__init__()
        # self.down_1 = nn.Sequential(
            # nn.Conv2d(1, 64, 5, stride=2, padding=2),
            # nn.ReLU(True),
            # nn.Conv2d(64, 128, 5, stride=2, padding=2),
            # nn.ReLU(True))
        # self.res_block_1 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.res_block_2 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.res_block_3 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.down_2 = nn.Sequential(
            # nn.Conv2d(128, 32, 5, stride=2, padding=2),       
            # nn.ReLU(True))
        # self.up_1 = nn.Sequential(
            # nn.Conv2d(32, 512, 3, stride=1, padding=1), 
            # nn.PixelShuffle(2),            
            # nn.ReLU(True))
        # self.res_block_4 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.res_block_5 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.res_block_6 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),       
            # nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(True))
        # self.up_2 = nn.Sequential(
            # nn.Conv2d(128, 256, 3, stride=1, padding=1), 
            # nn.PixelShuffle(2),            
            # nn.ReLU(True))
        # self.up_3 = nn.Sequential(
            # nn.Conv2d(64, 4, 3, stride=1, padding=1), 
            # nn.PixelShuffle(2),            
            # nn.Tanh())
        # self.quantizer = Quantizer()

    # def forward(self, x):
        # down = self.down_1(x)
        # res_1 = self.res_block_1(down)
        # down = down + res_1
        # res_2 = self.res_block_2(down)
        # down = down + res_2
        # res_3 = self.res_block_3(down)
        # down = down + res_3
        # x = self.down_2(down)
        # x = self.quantizer(x)
        # up = self.up_1(x)       
        # res_4 = self.res_block_4(up)
        # up = up + res_3
        # res_5 = self.res_block_5(up)
        # up = up + res_4
        # res_6 = self.res_block_6(up)
        # up = up + res_6
        # x = self.up_2(up)
        # x = self.up_3(x)
        # return x
        
def cae(**kwargs):
    model = CAE(**kwargs)
    return model
    