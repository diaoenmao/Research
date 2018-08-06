import torch
import _functions

def quantizer(input, inplace=False):
    return _functions.quantizer.Quantizer.apply(input, inplace)