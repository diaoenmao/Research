import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Organic(InplaceFunction):

    @staticmethod
    def _make_noise(input,z):
        return z.resize_(input.size(0), z.size(0),
                                   *repeat(1, input.dim() - 2))
                                   
    @classmethod
    def forward(cls, ctx, input, z, p, train=False, inplace=False):
        ctx.noise = cls._make_noise(input,z)
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if not ctx.train:
            return input

        if ctx.inplace:
            output = input
        else:
            output = input.clone()
        ctx.noise = ctx.noise.div_(ctx.p)
        output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.noise, None, None, None
        else:
            return grad_output, None, None, None
