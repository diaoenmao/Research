import torch
import _functions

def quantizer(input, inplace=False):
    return _functions.quantizer.Quantizer.apply(input, inplace)

def sign(input, inplace=False):
    return _functions.sign.Sign.apply(input, if_training)