import torch
import numpy as np

output_sizes = [(128,1,32,32)]
output_features = 10
def main():
    x = conv(output_sizes[0])
    print(output_sizes)
    return
    
def alexnet(x):
    x = conv2d(x,out_channels=64,kernel_size=11,stride=4,padding=5)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = conv2d(x,out_channels=192,kernel_size=3,stride=1)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = conv2d(x,out_channels=384,kernel_size=3,padding=1)
    x = conv2d(x,out_channels=256,kernel_size=3,padding=1)
    x = conv2d(x,out_channels=256,kernel_size=3,padding=1)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = flatten(x)
    x = linear(x,output_channels=4096)
    x = linear(x,output_channels=4096)
    x = linear(x,output_channels=output_features)
    return x

def conv(x):
    x = conv2d(x,out_channels=96,kernel_size=3,stride=1,padding=1)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = conv2d(x,out_channels=128,kernel_size=3,stride=1,padding=1)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = conv2d(x,out_channels=256,kernel_size=3,stride=1,padding=1)
    x = maxpool2d(x,kernel_size=2,stride=2)
    x = flatten(x)
    x = linear(x,output_channels=2048)
    x = linear(x,output_channels=2048)
    x = linear(x,output_channels=output_features)
    return x  
    
def conv2d(input_size, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    H_in,W_in = input_size[2],input_size[3]
    H_out = np.floor((H_in+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    W_out = np.floor((W_in+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    output_size = (input_size[0],out_channels,H_out,W_out)
    output_sizes.append(output_size)
    return output_size
    
def maxpool2d(input_size, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    if(stride is None):
        stride = kernel_size
    H_in,W_in = input_size[2],input_size[3]
    H_out = np.floor((H_in+2*padding-dilation*(kernel_size-1)-1)/stride+1).astype(int)
    W_out = np.floor((W_in+2*padding-dilation*(kernel_size-1)-1)/stride+1).astype(int)
    output_size = (input_size[0],input_size[1],H_out,W_out)
    output_sizes.append(output_size)
    return output_size

def linear(input_size, output_channels, bias=True):
    other_dims = list(input_size[1:-1])
    output_size = [input_size[0]]
    output_size.extend(other_dims)
    output_size.append(output_channels)
    output_size = tuple(output_size)
    output_sizes.append(output_size)
    return output_size

def flatten(input_size):
    input_size = list(input_size)
    for i in range(len(input_size)):
        input_size[i] = 1 if input_size[i]==0 else input_size[i]
    output_size = (input_size[0],np.prod(input_size[1:]).astype(int))
    output_sizes.append(output_size)
    return output_size    
    
if __name__ == "__main__":
    main()