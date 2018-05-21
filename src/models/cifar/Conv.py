import torch
import numpy as np
from torch import nn



class Conv(nn.Module):

    def __init__(self, num_classes=1000):
            super(Conv, self).__init__()
            self.features = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256*(2**4), 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
            
def conv(**kwargs):
    model = Conv(**kwargs)
    return model