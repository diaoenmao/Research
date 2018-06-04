import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Organic(InplaceFunction):

    @staticmethod
    def _make_noise(input,z): 
        assert input.size(0)==z.size(0) and input.size(1)==z.size(1)
        a = torch.reshape(z,(input.size(0), input.size(1), *repeat(1, input.dim() - 2)))
        return a
                                   
    @classmethod
    def forward(cls, ctx, input, z, p, train=False, inplace=False):
        ctx.p = p
        ctx.train = train
        if (ctx.p.dim() == 0):
            if(ctx.p.item() == 1 or not ctx.train):
                return input
        else:
            if((ctx.p == 1).all() or not ctx.train):
                return input 
        noise = cls._make_noise(input,z)         
        ctx.inplace = inplace              
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        if (ctx.p.dim() == 0):
            if(ctx.p == 0):
                ctx.noise = noise.fill(0)
            ctx.noise = noise.div(ctx.p)
        else:
            ctx.noise = noise.clone()
            ctx.noise[:,ctx.p==0,] = 0
            ctx.noise[:,ctx.p!=0,] = ctx.noise[:,ctx.p!=0,].div_(ctx.p[ctx.p!=0])
        ctx.noise = ctx.noise.expand_as(input)
        output.mul_(ctx.noise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None
