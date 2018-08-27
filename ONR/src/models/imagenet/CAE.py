import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Quantize, Sign

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

class Separable_Conv(nn.Module):
    def __init__(self,inchan, outchan, kernel_size=1, stride=1, padding=0, dilation=1):
        super(Separable_Conv,self).__init__()
        
        self.conv = nn.Conv2d(inchan, inchan, kernel_size, stride, padding, dilation, groups=inchan)
        self.pointwise = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.activation = Activation(outchan)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.activation(self.pointwise(x))
        return x
        
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
    def __init__(self, Conv, inchan, outchan, depth):
        super(DownTransition, self).__init__()
        self.downsample = DownSample(inchan, outchan)
        self.conv = _make_Conv(Conv,outchan,outchan,depth)
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
    def __init__(self, Conv, inchan, outchan, depth):
        super(UpTransition, self).__init__()
        self.upsample = UpSample(inchan,outchan)
        self.conv = _make_Conv(Conv,outchan,outchan,depth)
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
        self.quantize = Quantize()
        
    def forward(self,x):
        x = self.quantize(x)
        return x
        
class Binarizer(nn.Module):
    def __init__(self,inchan,outchan):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=1)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = torch.tanh(feat)
        return self.sign(x)
        
class Encoder(nn.Module):
    def __init__(self, Conv):
        super(Encoder, self).__init__()
        self.input = InputTransition(32)
        self.down_0 = DownTransition(Conv,32,64,1)
        self.down_1 = DownTransition(Conv,64,64,1)
        self.down_2 = DownTransition(Conv,64,64,1)    
        self.down_3 = DownTransition(Conv,64,32,1)        
        self.quantizer = Quantizer()

    def forward(self, x):
        x = self.input(x)
        x = self.down_0(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.quantizer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, Conv):
        super(Decoder, self).__init__()
        self.up_0 = UpTransition(Conv,32,64,1) 
        self.up_1 = UpTransition(Conv,64,64,1)
        self.up_2 = UpTransition(Conv,64,64,1)
        self.up_3 = UpTransition(Conv,64,32,1)
        self.output = OutputTransition(32)      

    def forward(self, x):
        x = self.up_0(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.output(x)
        return x
        
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder(Basic_Conv)
        self.quantizer = Quantizer()
        self.net = Net()
        self.decoder = Decoder(Basic_Conv)

    def compression_loss_fn(self,output,target):
        res = (output-target).abs().mean()
        return res 
        
    def forward(self, input):
        x = self.encoder(input)
        code = self.quantizer(x)
        output = self.net(code)
        image = self.decoder(x)
        compression_loss = self.compression_loss_fn(image,input)
        return compression_loss,image,output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            32, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,10)
        
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    