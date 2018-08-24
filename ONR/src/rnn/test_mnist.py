import numpy as np
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from data import *
from util import *
from scipy.misc import imread, imresize, imsave

randomGen = np.random.RandomState(2)
train_dataset,test_dataset = fetch_dataset(data_name='MNIST')
train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size=10,batch_size=1,num_fold=0,radomGen=randomGen)


img = list(train_loader)[0][0]
rgb_img = torch.cat((img,img,img),1)
imsave(
        './mnist.png',
        np.squeeze(rgb_img.numpy().clip(0, 1) * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))