import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Organic(InplaceFunction):

    @staticmethod
    def _make_noise(input,z): 
        assert input.size(0)==z.size(0) and input.size(1)==z.size(1)
        z = torch.reshape(z,(input.size(0), input.size(1), *repeat(1, input.dim() - 2))).to(input.device)
        return z
                                   
    @classmethod
    def forward(cls, ctx, input, z, p, train=False, inplace=False):
        ctx.p = p.to(input.device)
        ctx.train = train
        if (ctx.p.dim() == 0):
            if(ctx.p.item() == 1 or not ctx.train):
                return input
        else:
            if((ctx.p == 1).all() or not ctx.train):
                return input 
        ctx.noise = cls._make_noise(input,z)         
        ctx.inplace = inplace              
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
            
        ctx.noise = ctx.noise.div_(ctx.p)
        output.mul_(ctx.noise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None
