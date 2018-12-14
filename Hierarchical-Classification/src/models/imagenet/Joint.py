import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import copy
from modules import ConvLSTMCell, Sign
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB
from metrics import flatten_output
from models.baseline import *
config.init()
device = config.PARAM['device']
max_channel = config.PARAM['max_channel']

class ClassifierCell(nn.Module):
    def __init__(self, n_class):
        super(ClassifierCell, self).__init__()
        self.n_class = n_class
        self.fc = nn.Linear(max_channel, n_class)
        
    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, (1, 1)).view(input.size(0),-1)
        x = self.fc(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, class_size):
        super(Classifier, self).__init__()
        self.class_size = class_size
        self.cell, self.branch_classifiers = self.make_layers(class_size)
        
    def make_layers(self, class_size):
        if(isinstance(class_size, int)):
            cell = ClassifierCell(class_size)
            branch_classifiers = None
        elif(isinstance(class_size, list)):
            cell = ClassifierCell(len(class_size))
            branch_classifiers = self.make_classfiers_from_branch(class_size)
        else:
            raise ValueError('Not supported type making classifiers from branch')
        return cell, branch_classifiers
        
    def make_classfiers_from_branch(self, class_size):        
        branch_classifiers = nn.ModuleList([])
        for i in range(len(class_size)):
            branch_classifiers.append(Classifier(class_size[i]))
        return branch_classifiers
        
    def classification_loss_fn(self, output, target, protocol):
        if('classes_counts' in protocol):
            classes_counts = protocol['classes_counts']
            inverse_proportion = classes_counts.sum()/classes_counts
            classes_weight = inverse_proportion/inverse_proportion.sum()*self.cell.n_class
        else:
            classes_weight = None
        loss = F.nll_loss(output,target,weight=classes_weight)
        return loss
    
    def forward(self, input, protocol):      
        feature = input['feature']
        label = input['label']
        output = {'this':None,'child':None}
        
        output['this'] = self.cell(feature)
        
        if(self.branch_classifiers is None):
            return output
        elif(isinstance(self.branch_classifiers, nn.ModuleList)):
            output['child'] = []
            for i in range(len(self.branch_classifiers)):
                sub_output = self.branch_classifiers[i](input,protocol)
                output['child'].append(sub_output)
        else:
            raise ValueError('Not supported type forwarding classifier')
        return output

class Joint(nn.Module):
    def __init__(self,classes_size,pretrained=False):
        super(Joint, self).__init__()
        self.classes_size = classes_size
        self.feature_extractor = resnet18(pretrained=pretrained, if_classify=False)
        self.classifier = Classifier(self.classes_size)
        
    def loss_fn(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        loss = tuning_param['classification']*self.classifier.classification_loss_fn(output['classification'],input['label'],protocol)
        return loss

    def forward(self, input, protocol):
        output = {}
        
        feature = self.feature_extractor(input['img'],protocol)
        classification_output = self.classifier({'feature':feature, 'label': input['label']},protocol)
        flat_output = flatten_output(classification_output)
        if(self.classifier.branch_classifiers is not None):
            normalization_factor = torch.log(torch.exp(flat_output).sum(dim=1,keepdim=True))
            flat_output = flat_output - normalization_factor
        output['classification'] = flat_output
        output['loss'] = self.loss_fn(input,output,protocol)
        return output



        
    
    