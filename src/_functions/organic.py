import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Organic(InplaceFunction):

    @staticmethod
    def _make_noise(input,z): 
        assert input.size(1)==z.size(1)
        z = z.expand(input.size(0),input.size(1)).to(input.device)
        z = torch.reshape(z,(input.size(0), input.size(1), *repeat(1, input.dim() - 2)))
        return z
                                   
    @classmethod
    def forward(cls, ctx, input, z, p, train=False, inplace=False):
        ctx.noise = cls._make_noise(input,z)
        ctx.p = p.to(input.device)
        ctx.train = train
        ctx.inplace = inplace
        if (ctx.p.dim() == 0):
            if(ctx.p.item() == 1 or not ctx.train):
                return input
        else:
            if((ctx.p == 1).all() or not ctx.train):
                return input            
            
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
