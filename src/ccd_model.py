import math
import torch
from torch import nn
from torch.nn.parameter import Parameter



class ccd_Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features):
        super(ccd_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parameters = (self.in_features+1)*(self.out_features)
        for i in range(self.num_parameters):
            self.register_parameter(str(i), Parameter(torch.Tensor(1,))) 
        self.reset_parameters()
        #self.tmp = torch.cat(list(self.parameters()), dim=0)
        # self.p = [Parameter(torch.Tensor(1,)) for i in range(self.num_parameters)]
        # self.matrix = torch.cat(self.p, dim=0).view(out_features,self.in_features+1)
        # print(list(self.parameters()))
        # print(self.tmp)
        # print(self.weight)
        # print(self.bias)
        # self.reset_parameters()
        # print(list(self.parameters()))
        # print(self.tmp)
        # print(self.weight)
        # print(self.bias)
        # exit()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def forward(self, input):
        self.matrix = torch.cat(list(self.parameters()), dim=0).view(self.out_features,self.in_features+1)
        self.weight = self.matrix[:,:self.in_features]
        self.bias = self.matrix[:,-1]
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features)
        
        