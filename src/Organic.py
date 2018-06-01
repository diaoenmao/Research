from torch import nn
from functional import *
import time
from torch.distributions.bernoulli import Bernoulli   
from torch.distributions.beta import Beta
from matplotlib import pyplot as plt
     
class Organic(nn.Module):

    def __init__(self, in_channels, p=torch.tensor([0.5]), inplace=False):
        super(Organic, self).__init__()
        self.in_channels = in_channels
        if(p.dim()==0):            
            self.prior = torch.ones(2)*50
            self.concentration = torch.zeros(2)       
            self.p = p
            self.if_collapse = True
        else:
            self.prior = torch.ones(2,self.in_channels)*50
            self.concentration = torch.zeros(2,self.in_channels)
            self.p = torch.ones(in_channels)*p
            self.if_collapse = False
        self.z = Bernoulli(torch.ones(100,self.in_channels)*self.p).sample()
        self.info = Organic_info(self.prior,self.concentration,self.p)
        self.inplace = inplace
    
    def Beta(self):
        pos_concentration = self.prior+self.concentration
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
        
    def step(self,new_z):
        counts_1,counts_0 = self.count(new_z)
        if(self.if_collapse):
            self.concentration[0] = counts_1
            self.concentration[1] = counts_0
        else:
            self.concentration[0,:] = counts_1
            self.concentration[1,:] = counts_0

    def pred(self):
        self.prior += self.concentration
        if(self.if_collapse):
            self.concentration = torch.zeros(2)
        else:
            self.concentration = torch.zeros(2,self.in_channels)
            
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
        return organic(input, self.z, self.p, self.training, self.inplace)
        
        
        
class Organic_info:        
    def __init__(self,prior,concentration,p):
        self.prior = [prior]
        self.concentration = [concentration]
        self.p = [p]
        
    def update(self,prior=None,concentration=None,p=None):
        if(prior is not None):
            self.prior.append(prior)
        if(concentration is not None):
            self.concentration.append(concentration)
        if(p is not None):
            self.p.append(p)
            
def update_organic(mw,mode,input=None,target=None,data_loader=None):
    with torch.no_grad():
        for m in mw.model.modules():
            if isinstance(m, Organic):
                if(mode=='fast'):
                    fast_organic(input,target,mw,m)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                elif(mode=='mc'):
                    mc_organic(input,target,mw,m)
                elif(mode=='genetic'):
                    genetic_organic(data_loader,mw,m)
                elif(mode=='forget'):
                    forget_organic(m)
                elif(mode=='dropout'):
                    dropout(input,m)
    return

def report_organic(mw):
    info = []
    with torch.no_grad():
        for m in mw.model.modules():
            if isinstance(m, Organic):
                info.append(m.info)
    return info
    
def dropout(input,m):
    in_channels = m.in_channels
    p = m.p
    new_z = Bernoulli(torch.ones(input.size(0),in_channels)*p).sample()
    m.update(p=p,z=new_z)
    return
        
def fast_organic(input,target,mw,m):
    nsteps = 1
    for i in range(nsteps):
        #print(i)
        if(i==0):
            output = mw.model(input)
            current_likelihood = torch.exp(-mw.loss(output,target))
        new_beta = m.Beta()
        cur_z,cur_p = m.z,m.p
        new_p = new_beta.sample()
        new_z = Bernoulli(torch.ones(input.size(0),m.in_channels)*new_p).sample() 
        cur_counts_1,cur_counts_0 = m.count(cur_z)
        new_counts_1,new_counts_0 = m.count(new_z)
        cur_beta_jump = Beta(new_beta.concentration1+cur_counts_1,new_beta.concentration0+cur_counts_0)
        new_beta_jump = Beta(new_beta.concentration1+new_counts_1,new_beta.concentration0+new_counts_0)
        correction_numerator = torch.exp(new_beta_jump.log_prob(cur_p))
        correction_denominator = torch.exp(cur_beta_jump.log_prob(new_p))
        c = correction_numerator/correction_denominator
        # print(cur_counts_1)
        # print(cur_counts_0)
        # print(new_counts_1)
        # print(new_counts_0)
        # print(new_beta.concentration1+new_counts_1)
        # print(new_beta.concentration0+new_counts_0)
        # print(cur_p)
        # print(correction_numerator)
        # print(new_beta.concentration1+cur_counts_1)
        # print(new_beta.concentration0+cur_counts_0)
        # print(new_p)
        # print(correction_denominator)
        # print(c)
        # exit()
        
        m.update(p=new_p,z=new_z)        
        output = mw.model(input)
        new_likelihood = torch.exp(-mw.loss(output,target))
        accept_probability = torch.min(torch.tensor([1,(new_likelihood/current_likelihood).item() * c.item()]))
        u = torch.rand(1)

        # print(cur_p.item())
        # pt(new_p.item())
        # print(current_likelihood.item())
        # print(new_likelihood.item())
        # print(accept_probability.item())
        # print(u.item())
        if(accept_probability>=u):
            m.step(new_z)
            m.update(p=new_p,z=new_z)
            m.info.update(prior=m.prior,concentration=m.concentration,p=new_p)
            current_likelihood = new_likelihood
        else:
            m.step(cur_z)
            m.update(p=cur_p,z=cur_z)
            m.info.update(prior=m.prior,concentration=m.concentration,p=cur_p)
    m.pred()
    return

def forget_organic(m):
    m.prior /= 25000
    m.prior *= 50
    return
    
def mc_organic(data_loader,mw,m,device):
    p,z,samples = m.info.get_p(),m.info.get_z(),m.info.get_samples()
    new_p = p
    for i, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        m.update(p,z)
        output = mw.model(input)
        base_loss = mw.loss(output,target)
        for j in range(m.in_channels):
            flipped_z = z
            flipped_z[j] = 1 - flipped_z[j]
            m.update(p,flipped_z)
            output = mw.model(input)
            flipped_loss = mw.loss(output,target)            
            if(p.dim()==0):
                if(z[j]==0):
                    new_p_1 = new_p*flipped_loss
                    new_p_0 = (1-new_p)*loss
                    new_p = new_p_1/(new_p_1+new_p_0)
                if(z[j]==1):
                    new_p_1 = new_p*loss
                    new_p_0 = (1-new_p)*flipped_loss
                    new_p = new_p_1/(new_p_1+new_p_0)
            else:
                if(z[j]==0):
                    new_p_1 = new_p[j]*flipped_loss
                    new_p_0 = (1-new_p[j])*loss
                    new_p[j] = new_p_1/(new_p_1+new_p_0)
                if(z[j]==1):
                    new_p_1 = new_p[j]*loss
                    new_p_0 = (1-new_p[j])*flipped_loss
                    new_p[j] = new_p_1/(new_p_1+new_p_0)
            
    new_z = torch.bernoulli(torch.ones(1,m.in_channels)*new_p).to(torch.uint8)
    m.update(new_p,new_z)
    m.info.update(new_p,new_z)
    m.info.record(new_z)
    return  
    
    
def genetic_organic(data_loader,mw,m,device):
    min_population_size = 1000
    p,z,samples = m.info.get_p(),m.info.get_z(),m.info.get_samples()
    if(samples.size(0)<min_population_size):
        samples = torch.cat((samples,torch.bernoulli(torch.ones(min_population_size-samples.size(0),min_channels)*p)),0).to(torch.uint8)
    for i, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        p,z,samples = m.info.get_p(),m.info.get_z(),m.info.get_samples()           
        sorted_fitness, indices = fitness(input,target,samples)
        selected = select(indices)
        breeders = samples[indices,:]
        children = crossover(breeders)
        mutated_children = mutate(children)
        samples[indices[-muted_children.size(0):],] = mutated_children
        if(p.dim()==0):
            new_p = torch.mean(samples)
        else:
            new_p = torch.mean(samples,0)
        new_z = samples[indices[0],:]
        new_samples = samples
        m.update(new_p,new_z)
        m.info.update(new_p,new_z)
        m.info.record(new_samples)
    return
        
                
def fitness(input,target,samples):
    num_samples = samples.size(0)
    fitness = torch.zeros(num_samples)
    for i in range(num_samples):
        m.update(p,samples[i,:])
        output = mw.model(input)
        fitness[i] = mw.loss(output,target)
    sorted_fitness, indices = torch.sort(fitness,descending = True)
    return sorted_fitness, indices

def select(indices,mode='truncation'):
    if(mode=='truncation'):
        num_best_sample = 200
        num_lucky_samples = 200
        selected = indices[:num_best_sample]
        perm = torch.randperm(indices.size(0))
        selected.cat(indices[perm[:num_lucky_samples]])
    return selected    
        
def crossover(breeders):
    num_children = 2
    perm = torch.randperm(breeders.size(0))            
    breeders = breeders[perm,:]
    num_parents = int(breeders.size(0)/2)
    children = torch.zeros(num_children*num_parents,breeders.size(1))
    for i in range(num_parents):
        parents = samples[[selected_nextGen[i]],selected_nextGen[selected_nextGen.size(0)-1-i],:]
        children[i*num_children:(i+1)*num_children,:] = reproduce(num_children,parents)                        
    return children  


def reproduce(num_children,parents):
    children = torch.zeros(num_children,parents[0].size(0))
    pointer_samples = torch.rand(parents[0].size(0))
    pointers = torch.nonzero(pointer_samples)
    num_pointers = torch.sum(pointer_samples)
    swap_samples = torch.rand(num_children,num_pointers)
    start_pointer = 0
    for i in range(len(pointer_samples.size(0))):
        for j in range(num_children):
            children[j,start_pointer:pointer_samples[i]] = parents[swap_samples[j,i],start_pointer:pointer_samples[i]]
    return children
            
def mutate(children):
    mutation_rate = torch.ones(children.size())*(1/children.size(1))
    mutate_idx = torch.nonzero(torch.bernoulli(mutation_rate))
    children[mutate_idx] = 1 - children[mutate_idx]
    return children
    
    



    