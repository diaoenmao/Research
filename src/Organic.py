from torch import nn
from functional import *
import time
from torch.distributions.bernoulli import Bernoulli   
from torch.distributions.beta import Beta
from matplotlib import pyplot as plt
     
class Organic(nn.Module):

    def __init__(self, in_channels, p=torch.tensor([0.5]), device='cuda:0', inplace=False):
        super(Organic, self).__init__()
        self.in_channels = in_channels
        self.device = device
        if(p.dim()==0):            
            self.prior = torch.ones(2)*50
            self.concentration = torch.zeros(2)       
            self.p = p.to(self.device)
            self.if_collapse = True
        else:
            self.prior = torch.ones(2,self.in_channels)*50
            self.concentration = torch.zeros(2,self.in_channels)
            self.p = torch.ones(in_channels,device=self.device)*p.to(self.device)
            self.if_collapse = False
        self.z = Bernoulli(torch.ones(self.in_channels,device=self.device)*self.p).sample((100,))
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
        
    def step(self,z):
        counts_1,counts_0 = self.count(z)
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
        self.p = [p.to('cpu')]
        
    def update(self,prior=None,concentration=None,p=None):
        if(prior is not None):
            self.prior.append(prior)
        if(concentration is not None):
            self.concentration.append(concentration)
        if(p is not None):
            self.p.append(p.to('cpu'))
            
def update_organic(mw,mode,input=None,target=None,data_loader=None):
    with torch.no_grad():
        for m in mw.model.modules():
            if isinstance(m, Organic):
                if(mode=='mh'):
                    mh_organic(input,target,mw,m)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                elif(mode=='gibbs'):
                    gibbs_organic(input,target,mw,m)
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
    new_z = Bernoulli(torch.ones(in_channels)*p).sample((input.size(0),))
    m.update(p=p,z=new_z)
    return
        
def mh_organic(input,target,mw,m):
    nsteps = 1
    for i in range(nsteps):
        #print(i)
        cur_beta = m.Beta()
        cur_p = cur_beta.sample().to(m.device)
        cur_ber = Bernoulli(torch.ones(m.in_channels,device=m.device)*cur_p) 
 
        cur_z = cur_ber.sample((input.size(0),))
        m.update(p=cur_p,z=cur_z)
        output = mw.model(input)
        cur_likelihood = -mw.loss(output,target)
        if(m.if_collapse):
            cur_prior_likelihood = torch.mean(cur_ber.log_prob(cur_z))
        else:
            cur_prior_likelihood = torch.mean(cur_ber.log_prob(cur_z),dim=0)
        cur_pos = cur_likelihood+cur_prior_likelihood

        
        opposite_cur_z = 1-cur_z        
        m.update(p=cur_p,z=opposite_cur_z)
        opposite_cur_output = mw.model(input)
        opposite_cur_likelihood = -mw.loss(opposite_cur_output,target)
        if(m.if_collapse):
            opposite_cur_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_cur_z))
        else:
            opposite_cur_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_cur_z),dim=0)
        opposite_cur_pos = opposite_cur_likelihood+opposite_cur_prior_likelihood      
        proposal_cur_p = 1/(1+torch.exp(opposite_cur_pos-cur_pos))
        proposal_cur_ber = Bernoulli(torch.ones(m.in_channels,device=m.device)*proposal_cur_p) 
        
        new_z = proposal_cur_ber.sample((input.size(0),))
        m.update(p=cur_p,z=new_z)
        output = mw.model(input)
        new_likelihood = -mw.loss(output,target)
        if(m.if_collapse):
            new_prior_likelihood = torch.mean(cur_ber.log_prob(new_z))
        else:
            new_prior_likelihood = torch.mean(cur_ber.log_prob(new_z),dim=0)
        new_pos = new_likelihood+new_prior_likelihood
        opposite_new_z = 1-new_z
        m.update(p=cur_p,z=opposite_new_z)
        opposite_new_output = mw.model(input)
        opposite_new_likelihood = -mw.loss(opposite_new_output,target)
        if(m.if_collapse):
            opposite_new_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_new_z))
        else:
            opposite_new_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_new_z),dim=0)
        opposite_new_pos = opposite_new_likelihood+opposite_new_prior_likelihood  
        proposal_new_p = 1/(1+torch.exp(opposite_new_pos-new_pos))
        proposal_new_ber = Bernoulli(torch.ones(m.in_channels,device=m.device)*proposal_new_p)
       
        new_to_cur = torch.mean(proposal_new_ber.log_prob(cur_z))
        cur_to_new = torch.mean(proposal_cur_ber.log_prob(new_z))
        accept_probability = torch.min(torch.exp(torch.mean(new_pos) - torch.mean(cur_pos) + new_to_cur - cur_to_new),torch.tensor(1.0,device=m.device))

        u = torch.rand(1,device=input.device)

        if(accept_probability>=u):
            m.step(new_z)
            m.update(p=cur_p,z=new_z)
            m.info.update(prior=m.prior,concentration=m.concentration,p=cur_p)
        else:
            m.step(cur_z)
            m.update(p=cur_p,z=cur_z)
            m.info.update(prior=m.prior,concentration=m.concentration,p=cur_p)
    m.pred()
    return

def forget_organic(m):
    m.prior /= 25000
    m.prior *= 50
    m.prior += 50
    return
    
def gibbs_organic(input,target,mw,m):
    cur_p = m.p
    cur_ber = Bernoulli(torch.ones(m.in_channels,device=m.device)*cur_p)
    
    cur_z = m.z
    m.update(p=cur_p,z=cur_z)
    output = mw.model(input)
    cur_likelihood = -mw.loss(output,target)
    if(m.if_collapse):
        cur_prior_likelihood = torch.mean(cur_ber.log_prob(cur_z))
    else:
        cur_prior_likelihood = torch.mean(cur_ber.log_prob(cur_z),dim=0)
    cur_pos = cur_prior_likelihood+cur_likelihood      
    opposite_cur_z = 1 - cur_z
    m.update(p=cur_p,z=opposite_cur_z)
    opposite_output = mw.model(input)
    opposite_likelihood = -mw.loss(opposite_output,target) 
    if(m.if_collapse):
        opposite_cur_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_cur_z))
    else:
        opposite_cur_prior_likelihood = torch.mean(cur_ber.log_prob(opposite_cur_z),dim=0)
    opposite_pos = opposite_cur_prior_likelihood+opposite_likelihood
    new_p = 1/(1+torch.exp(opposite_pos-cur_pos)) 
    new_z = Bernoulli(torch.ones(m.in_channels,device=m.device)*new_p).sample((input.size(0),)) 
    m.update(p=new_p,z=new_z)
    m.info.update(p=new_p)
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
    
    



    