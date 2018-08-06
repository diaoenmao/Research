import torch
from torch.autograd.function import Function

class Quantizer(Function):
    def forward(self, input):
        output = torch.round(input)
        return output

    def backward(self, grad_output):
        return grad_output 