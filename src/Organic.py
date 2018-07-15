from torch import nn
from functional import *
import time
import config
from torch.distributions.bernoulli import Bernoulli   
from torch.distributions.beta import Beta
from matplotlib import pyplot as plt

config.init()
    
class Organic(nn.Module):

    def __init__(self, in_channels, p=torch.tensor([0.5]), device=config.PARAM['device'], inplace=False):
        super(Organic, self).__init__()
        self.in_channels = in_channels
        self.device = device
        #self.tracker = 0
        if(p.dim()==0):            
            #self.prior = torch.ones(2)*config.PARAM['batch_size']/2
            #self.concentration = torch.zeros(2)
            #self.p = torch.ones(config.PARAM['data_size'],1,device=self.device)*p.to(self.device)
            #self.z = Bernoulli(torch.ones(config.PARAM['data_size'],self.in_channels,device=self.device)*self.p).sample()
            self.p = p.to(self.device)
            self.if_collapse = True
        else:
            # self.prior = torch.zeros(int(config.PARAM['data_size']/config.PARAM['batch_size']),2,self.in_channels)
            # self.prior[0,:,:] = torch.ones(2,self.in_channels)*config.PARAM['batch_size']/2
            # self.prior_tracker = 0
            # self.concentration = torch.zeros(2,self.in_channels)
            self.p = torch.ones(in_channels,device=self.device)*p.to(self.device)
            self.if_collapse = False
        self.z = Bernoulli(torch.ones(self.in_channels,device=self.device)*self.p).sample((config.PARAM['batch_size'],))
        self.info = Organic_info(self.p,self.if_collapse)
        self.inplace = inplace
    
    def Beta(self):
        pos_concentration = self.prior[self.prior_tracker,:]+self.concentration
        if(self.if_collapse):            
            return Beta(pos_concentration[0],pos_concentration[1])
        else:
            return Beta(pos_concentration[0,:],pos_concentration[1,:])

    def count(self,z):
        if(self.if_collapse):
            counts_1 = torch.sum(torch.mean(z,dim=1))
            counts_0 = torch.sum(1-torch.mean(z,dim=1))
        else:
            counts_1 = torch.sum(z,dim=0)
            counts_0 = z.size(0)-counts_1
        return counts_1,counts_0
        
    def step(self,z):
        counts_1,counts_0 = self.count(z)
        if(self.if_collapse):
            self.concentration[0] = counts_1
            self.concentration[1] = counts_0
        else:
            self.concentration[0,:] = counts_1
            self.concentration[1,:] = counts_0

    def pred(self):
        # if(self.if_collapse):
            # self.prior[self.prior_tracker,:] += self.concentration
        # else:
            # self.prior[self.prior_tracker,:,:] = self.concentration
        if(self.prior_tracker==self.prior.size(0)-1):
            self.prior_tracker = 0 
        else:
            self.prior_tracker += 1
        # if(self.if_collapse):
            # self.concentration = torch.zeros(2)
        # else:
            # self.concentration = torch.zeros(2,self.in_channels)
  
    def update(self,prior=None,concentration=None,p=None,z=None):    
        if(prior is not None):
            self.prior = prior
        if(concentration is not None):
            self.concentration = concentration
        if(p is not None):
            self.p = p
        if(z is not None):
            self.z = z
        assert self.in_channels == self.z.size(1)
        assert (p>=0).all() and (p<=1).all()
        
    def forward(self, input):

    



    