import torch
from torch.autograd.function import Function

class Quantizer(Function):
    
    @staticmethod
    def forward(ctx, input, inplace):
        if(inplace):
            output = input.round_()
        else:
            output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None 