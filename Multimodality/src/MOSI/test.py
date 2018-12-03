import torch
import numpy as np
from util import *
import pickle


data_dir = './data/MOSI.pkl'
with open(data_dir, 'rb') as f:
    d = pickle.load(f) 
print(d)

