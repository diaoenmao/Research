import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import models
import os
from utils import *
from torchvision.utils import make_grid
from data import *
from PIL import Image


if __name__ == '__main__':
    batch_size = 10
    train_dataset, test_dataset = fetch_dataset('CIFAR100')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, num_workers=0,collate_fn=input_collate)
    print(len(train_dataset))    
    for i, input in enumerate(train_loader):
        print(input)
        exit()