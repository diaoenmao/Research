import numpy as np
from util import *

def early_stopping(loss,min_delta=1e-2, patience=10):
    best_epoch_id = 0
    best_epoch_loss = np.inf
    patience_tracker = 0
    for t in range(loss.shape[0]):
        if((loss[t]-best_epoch_loss)<-min_delta):
            best_epoch_id = t
            best_epoch_loss = loss[t]
            patience_tracker = 0
        else:
            if(patience_tracker>=patience):
                return best_epoch_id
            patience_tracker = patience_tracker + 1
    return t
    
def vectorize_parameters(param):
    vec_params = []
    for p in param:
        vec_params.append(p.contiguous().view(-1,1))
    vec_params = torch.cat(vec_params,dim=0)
    return vec_params
    
def get_AIC(dataSize,mw):
    num_param = mw.num_free_parameters()
    AIC = (num_param)/dataSize
    return AIC

def get_BIC(dataSize,mw):
    num_param = mw.num_free_parameters()
    BIC = (num_param*np.log(dataSize))/(2*dataSize)
    return BIC

def get_BC(dataSize,mws,loss,coef=2/3):
    BC = np.zeros(len(mws))
    AIC = regularization(dataSize,mws,loss,'AIC')
    AIC_selected = np.argmin(AIC)
    AIC_selected_num_parameters = mws[AIC_selected].num_free_parameters()
    for i in range(len(mws)):
        num_param = mws[i].num_free_parameters()
        if(num_param<=AIC_selected_num_parameters):
            adjustment = 0
            for j in range(1,num_param+1):
                adjustment = adjustment + 1/j
            BC[i] = torch.mean(loss[i])+((dataSize[i]**coef*adjustment)/(2*dataSize[i]))
        else:
            BC[i] = np.inf
    return BC
    
def get_GTIC(dataSize,mw,loss_batch):
    likelihood_batch = -loss_batch
    l_vec_free_grad_params=[]
    for j in range(np.int(dataSize)):
        grad_params = torch.autograd.grad(likelihood_batch[j], mw.parameters(), create_graph=True, only_inputs=True)
        tmp_free_grad_params = mw.free_parameters(list(grad_params))
        vec_free_grad_params = vectorize_parameters(tmp_free_grad_params)
        vec_free_grad_params = vec_free_grad_params.unsqueeze(0)
        l_vec_free_grad_params.append(vec_free_grad_params)         
    free_grad_params = torch.cat(l_vec_free_grad_params,dim=0)
    sum_free_grad_params = torch.sum(free_grad_params,dim=0)
    #print(sum_free_grad_params)
    non_zero_idx = torch.nonzero(sum_free_grad_params[:,0])
    non_zero_idx = non_zero_idx.data
    free_grad_params_T = free_grad_params.transpose(1,2)
    J_batch = torch.matmul(free_grad_params,free_grad_params_T)
    J = torch.sum(J_batch,dim=0)
    J = J[non_zero_idx,non_zero_idx.view(1,-1)]   
    J = J/dataSize
    #print('J')
    #print(J)
    H = []
    for j in sum_free_grad_params:
        h = torch.autograd.grad(j, mw.parameters(), create_graph=True)
        h = vectorize_parameters(h).view(1,-1)   
        H.append(h)
    H = torch.cat(H,dim=0)
    free_vec_parameters_idx = mw.free_vec_parameters_idx()
    H = H[:,free_vec_parameters_idx]
    H = H[non_zero_idx,non_zero_idx.view(1,-1)]
    #print('H')
    #print(H)
    V = -H/dataSize
    try:
        inv_V = torch.inverse(V)
        #print(inv_V)
        VmJ = torch.matmul(inv_V,J)
        tVMJ = torch.trace(VmJ)
        print('effective num of paramters')
        print(tVMJ.data[0])
        GTIC = tVMJ/dataSize
        if(GTIC.data[0]<0 or np.isnan(tVMJ.data[0])):
            print('numerically unstable')
            # print(J)
            # print(H)
            # print(inv_V)
            GTIC = 65535
    except RuntimeError as e:
        print('numerically unstable, not invertable')
        # print(e)
        # print(J)
        # print(H)
        GTIC = 65535
    #exit()
    return GTIC

def get_GTIC_approx(dataSize,mw,loss_batch):
    likelihood_batch = -loss_batch
    free_vec_parameters_idx = mw.free_vec_parameters_idx()
    params = mw.parameters()
    free_params = mw.free_parameters(params)
    l_approx=[]
    for i in range(np.int(dataSize)):
        grad_params = torch.autograd.grad(likelihood_batch[i], mw.parameters(), create_graph=True, only_inputs=True)
        tmp_free_grad_params = mw.free_parameters(grad_params)
        vec_free_grad_params = vectorize_parameters(tmp_free_grad_params)
        second_grad_params = []
        for j in vec_free_grad_params:
            h = torch.autograd.grad(j, mw.parameters(), create_graph=True)
            h = vectorize_parameters(h).view(1,-1)   
            second_grad_params.append(h)
        second_grad_params = torch.cat(second_grad_params,dim=0)    
        tmp_free_second_grad_params = second_grad_params[:,free_vec_parameters_idx]        
        cholesky_free_second_grad_params = torch.potrf(tmp_free_second_grad_params)
        approx = cholesky_free_second_grad_params.matmul(free_params)
        approx = approx.unsqueeze(0)
        l_approx.append(approx)        
    approx = torch.cat(l_approx,dim=0)
    tVMJ =  torch.var(approx,dim=0)
    GTIC = tVMJ/dataSize
    return GTIC    

def get_REG(dataSize,mw,loss_batch,regularization,regularization_param):
    reg = None
    if(regularization_param is not None):    
        for W in mw.model.parameters():
            for i in range(1,regularization_param.size()[0]+1):
                if reg is None:
                    reg = regularization_param[i-1] * W.norm(np.float(i))
                else:
                    reg = reg + regularization_param[i-1] * W.norm(np.float(i))
    if (regularization[0]!=0):
        reg = reg + get_GTIC(dataSize,mw,loss_batch)
    print(reg)
    print(regularization_param)
    exit()
    REG = reg
    return REG

def regularization(dataSizes,mws,loss_batches,mode):
    REG = np.zeros(len(mws))
    if(mode=='BC'):
        REG = get_BC(dataSizes,mws,loss_batches)  
        return REG
    for i in range(len(mws)):
        print(i)
        if(mode=='Base'):
            REG[i] = torch.mean(loss_batches[i])
        elif(mode=='AIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_AIC(dataSizes[i],mws[i])            
        elif(mode=='BIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_BIC(dataSizes[i],mws[i])  
        elif(mode=='GTIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_GTIC(dataSizes[i],mws[i],loss_batches[i])
        elif(mode=='REG'):
            REG[i] = torch.mean(loss_batches[i]) + get_REG(dataSizes[i],mws[i],loss_batches[i],mws[i].regularization,mws[i].regularization_parameters)
        else:
            print('mode not supported for model selection')
            exit()
    return REG
    
def get_GTIC_closed(input, target, model):
    X = input.data.cpu().numpy()
    y = target.data.cpu().numpy()
    beta = list(model.parameters())[0].t().data.cpu().numpy()
    bias = list(model.parameters())[1].data.cpu().numpy()
    mu = X.dot(beta)+ bias
    mu_1 = mu[:,0]
    mu_2 = mu[:,1]
    X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    m = beta.shape[0]+1
    n = y.shape[0]
    J = np.zeros((m,m))
    J_1 = np.zeros((m,1))
    J_2 = np.zeros((m,1))
    for i in range(n):
        J += (1-y[i] - np.exp(mu_1[i])/(np.exp(mu_1[i])+np.exp(mu_2[i])))**2 * X[i,:].reshape((m,1)).dot(X[i,:].reshape((1,m))) 
    J /= n
    H = np.zeros((m,m))
    for i in range(n): 
        H += -(np.exp(mu_1[i])*np.exp(mu_2[i])) / ((np.exp(mu_1[i])+np.exp(mu_2[i]))**2) * X[i,:].reshape((m,1)).dot(X[i,:].reshape((1,m)))   
    V = -H/n 
    pen = 2*np.trace(np.linalg.inv(V).dot(J))
    print(pen)
    return pen
    
    