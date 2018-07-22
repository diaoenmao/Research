import numpy as np
import config
from util import *
config.init()
max_num_epochs = config.PARAM['max_num_epochs']
device = torch.device(config.PARAM['device'])
            
def penalize(irreduced_loss,mw,mode,TAG):
    reduced_loss = torch.mean(irreduced_loss)
    if(mode=='AIC'):
        penalty = AIC(irreduced_loss.size(0),mw)
        penalized_loss =  reduced_loss + penalty
    elif(mode=='BIC'):
        penalty = BIC(irreduced_loss.size(0),mw)
        penalized_loss = reduced_loss + penalty
    elif(mode=='GTIC'):
        penalty = GTIC(irreduced_loss,mw)
        penalized_loss = reduced_loss + penalty
    print(penalty*irreduced_loss.size(0))
    return penalized_loss
    
def vectorize_parameters(param):
    vec_params = []
    for p in param:
        vec_params.append(p.reshape((-1)))
    vec_params = torch.cat(vec_params,dim=0)
    return vec_params
    
def AIC(dataSize,mw):
    num_free_parameters = mw.num_free_parameters()
    AIC = (num_free_parameters)/dataSize
    return AIC

def BIC(dataSize,mw):
    num_free_parameters = mw.num_free_parameters()
    BIC = (num_free_parameters*np.log(dataSize))/(2*dataSize)
    return BIC
    
def GTIC(irreduced_loss,mw):
    irreduced_likelihood = -irreduced_loss
    dataSize = irreduced_loss.size(0)
    num_free_parameters = mw.num_free_parameters()
    irreduced_grad_free_parameters = torch.zeros((dataSize,num_free_parameters),device=device)  
    for i in range(int(dataSize)):
        cur_grad_parameters = torch.autograd.grad(irreduced_likelihood[i], mw.parameters(), create_graph=True, only_inputs=True)
        cur_grad_free_parameters = mw.free_parameters(cur_grad_parameters)
        irreduced_grad_free_parameters[i,:] = vectorize_parameters(cur_grad_free_parameters)     
    J = torch.matmul(irreduced_grad_free_parameters.t(),irreduced_grad_free_parameters)/dataSize
    J = J + torch.eye(num_free_parameters)*1e-5
    # Eigenvalues = torch.eig(J)[0][:,0]
    # print(torch.min(Eigenvalues))
    likelihood = torch.mean(irreduced_likelihood)
    grad_parameters = torch.autograd.grad(likelihood, mw.parameters(), create_graph=True, only_inputs=True)
    grad_free_parameters = mw.free_parameters(grad_parameters)
    vec_grad_free_parameters = vectorize_parameters(grad_free_parameters)
    H = torch.zeros((num_free_parameters,num_free_parameters),device=device)
    for i in range(num_free_parameters):
        cur_grad2_parameters = torch.autograd.grad(vec_grad_free_parameters[i], mw.parameters(), retain_graph=True, only_inputs=True)
        cur_grad2_free_parameters = mw.free_parameters(cur_grad2_parameters)
        H[i,:] = vectorize_parameters(cur_grad2_free_parameters)
    V = -H
    try:
        inv_V = torch.inverse(V+torch.eye(num_free_parameters)*1e-5)
        #inv_V = torch.inverse(V)
    except RuntimeError as e:
        print(e)
        print('ill-conditioned V')
        return AIC(irreduced_loss.size(0),mw)
    VmJ = torch.matmul(inv_V,J)
    tVMJ = torch.trace(VmJ)
    out = tVMJ/dataSize
    return out
    
def GTIC_closed(input,output,target,mw):
    with torch.no_grad():
        num_free_parameters = mw.num_free_parameters()
        dataSize = target.size(0)
        input = torch.cat((input,torch.ones(dataSize,1)),dim=1)
        J = torch.zeros((num_free_parameters,num_free_parameters))
        for i in range(int(dataSize)):
            J += (1-target[i].float() - torch.exp(output[i,0])/(torch.exp(output[i,0])+torch.exp(output[i,1])))**2 * input[i,:].reshape((num_free_parameters,1)).matmul(input[i,:].reshape((1,num_free_parameters))) 
        J /= dataSize
        #J = J + torch.eye(num_free_parameters)*1e-5
        H = torch.zeros((num_free_parameters,num_free_parameters))
        for i in range(int(dataSize)): 
            H += -(torch.exp(output[i,0])*torch.exp(output[i,1])) / ((torch.exp(output[i,0])+torch.exp(output[i,1]))**2) * input[i,:].reshape((num_free_parameters,1)).matmul(input[i,:].reshape((1,num_free_parameters))) 
        V = -H/dataSize
        #inv_V = torch.inverse(V+torch.eye(num_free_parameters)*1e-5)
        inv_V = torch.inverse(V)
        VmJ = torch.matmul(inv_V,J)
        tVMJ = torch.trace(VmJ)
        out = tVMJ/dataSize
    return out  