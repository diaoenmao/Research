import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .utils import default_loader, make_dataset, merge_classes, make_classes_counts
IMG_EXTENSIONS = ['.bmp']

class WheatImage(Dataset):
    splits = ['binary', 'six']
    binary_classes = ['abnormal','normal']
    six_classes = [
                'cracked',
                'germinant',
                'moldy',
                'mothy',
                'normal',          
                'sick',               
                ]
                
    def __init__(self, root, split, transform=None):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(split, ', '.join(self.splits)))
        self.root = root
        self.split = split
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        if(self.split == 'binary'):
            self.classes = self.binary_classes
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            samples = make_dataset(self.root, self.extensions, self.classes_to_labels) 
            samples = merge_classes(samples, {'0':0,'1':0,'2':0,'3':0,'4':1,'5':0})
        elif(self.split == 'six'):
            self.classes = self.six_classes
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            samples = make_dataset(self.root, self.extensions, self.classes_to_labels)
        self.classes_counts = make_classes_counts(samples,self.classes_size)
        self.samples = samples
        self.transform = transform
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        label = torch.tensor(label)
        img = self.loader(path)
        input = {'img': img, 'label': label}
        if self.transform is not None:
            input = self.transform(input)            
        return input

    def __len__(self):
        return len(self.samples)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
