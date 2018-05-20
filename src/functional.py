import torch
from . import _functions

def organic(input, z, p, training=False, inplace=False):
    return _functions.organic.Organic.apply(input, z, p, training, inplace)