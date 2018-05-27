from torch import nn
from functional import *
import time
        
class Organic(nn.Module):

    def __init__(self, in_channels, p=torch.tensor(0.5), inplace=False):
        super(Organic, self).__init__()
        self.in_channels = in_channels
        if(p.dim()>0):
            p = torch.ones(in_channels)*p
        self.p = p
        self.z = torch.bernoulli(torch.ones(100,self.in_channels)*self.p)
        self.info = Organic_info(self.p,self.in_channels)
        self.inplace = inplace
        
    def update(self,p,z):
        self.p = p
        self.z = z
        self.info.update(self.p,self.z)
        assert self.in_channels == self.z.size(1)
        assert (p>=0).all() and (p<=1).all()
        
    def forward(self, input):
        return organic(input, self.z, self.p, self.training, self.inplace)
        
        
        
class Organic_info:        
    def __init__(self,p,in_channels):
        self.p = [p]
        self.window_size = 50000
        self.samples = torch.zeros(self.window_size,in_channels)
        self.sample_tracker = 0
        self.if_full = False
        
    def update(self,new_p,new_z): 
        self.p.append(new_p)
        num_samples = new_z.size(0)
        if(self.sample_tracker+num_samples<=self.window_size):
            new_tracker = self.sample_tracker+num_samples
            self.samples[self.sample_tracker:new_tracker,:] = new_z    
            self.sample_tracker = new_tracker
        else:
            num_extra_samples = self.sample_tracker+num_samples-self.window_size
            self.samples[self.sample_tracker:,:] = new_z[:-num_extra_samples,:]
            self.samples[:num_extra_samples,:] = new_z[-num_extra_samples:,:]
            new_tracker = num_extra_samples

        if(self.sample_tracker==self.window_size):
            self.sample_tracker = 0
            self.if_full = True
            
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
                elif(mode=='dropout'):
                    dropout(input,m)
    return

def report_organic(mw):
    p = []
    with torch.no_grad():
        for m in mw.model.modules():
            if isinstance(m, Organic):
                p.append(m.info.p)
    return p
    
def dropout(input,m):
    in_channels = m.in_channels
    p = m.p
    new_z = torch.bernoulli(torch.ones(input.size(0),in_channels)*p)
    m.update(p,new_z)
    return
        
def fast_organic(input,target,mw,m):
    nsteps = 1
    p_update_window_size = 100
    #s1 = time.time()
    for i in range(nsteps):
        if(m.info.if_full):
            if(i==0):
                output = mw.model(input)
                loss_tracker = mw.loss(output,target)
            sample_tracker = m.info.sample_tracker            
            if(sample_tracker<p_update_window_size):
                if(sample_tracker==0):
                    tmp_samples = m.info.samples[-p_update_window_size:,:]
                else:                    
                    tmp_samples = torch.cat((m.info.samples[:sample_tracker,:],m.info.samples[sample_tracker-p_update_window_size:,:]),dim=0)                    
                if(m.p.dim()==0):
                    new_p = torch.mean(tmp_samples)
                else:     
                    new_p = torch.mean(tmp_samples,dim=0)
            else:           
                if(m.p.dim()==0):
                    new_p = torch.mean(m.info.samples[sample_tracker-p_update_window_size:sample_tracker,:])
                else:          
                    new_p = torch.mean(m.info.samples[sample_tracker-p_update_window_size:sample_tracker:,:],dim=0)
            # print(new_p)
            # print((new_p==0).sum())
            new_z = torch.bernoulli(torch.ones(input.size(0),m.in_channels)*new_p)
            #print(new_z[:20,:])            
            m.update(new_p,new_z)
        else:
            new_p = m.p
            new_z = torch.bernoulli(torch.ones(input.size(0),m.in_channels)*new_p)
            m.update(new_p,new_z) 
            return        
        output = mw.model(input)
        new_loss = mw.loss(output,target)
        log_ratio = -new_loss+loss_tracker
        log_u = torch.log(torch.rand(1,device=log_ratio.device))
        # print(loss_tracker.item())
        # print(new_loss.item())
        # print(log_ratio.item())
        # print(log_u.item())
        if(log_ratio>log_u):
            # print('accept')
            # print(new_p)
            m.update(new_p,new_z)
            loss_tracker = new_loss
        else:
            # print('reject')
            # print(m.p)
            m.update(m.p,m.z)
        #e1 = time.time()
        #print(e1-s1)
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
    
    



    