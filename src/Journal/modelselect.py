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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def vectorize_parameters2(params):
    vec_params = []
    for p in params:
        print(p)
        print(p.contiguous().view(-1,1))
        vec_params.append(p.view(-1,1))
    print('ccc')
    vec_params = torch.cat(vec_params,dim=0)
    print('kkk')
    return vec_params
    
def vectorize_parameters(params):
    vec_params = []
    for p in params:
        vec_params.append(p.contiguous().view(-1,1))
    vec_params = torch.cat(vec_params,dim=0)
    return vec_params
    
def filter_free_parameters(param):
    free_param = []
    for p in param:
        if(p.size()[0]>1):
            free_param.append(p[:-1,])
        else:
            free_param.append(p)
    return free_param

def get_free_parameters(model):
    if(not model.ifclassification):
        return model.parameters()
    param = list(model.parameters())
    outputlayer_param = list(model.outputlayer.parameters())
    free_param = param[:-len(outputlayer_param)]
    outputlayer_free_param = filter_free_parameters(outputlayer_param)
    free_param.extend(outputlayer_free_param)
    return free_param
    
def get_free_grad_parameters(grad_parameters,model):
    grad_parameters = list(grad_parameters)
    if(not model.ifclassification):
        return grad_parameters   
    num_outputlayer_param = len(list(model.outputlayer.parameters()))
    outputlayer_grad_param = grad_parameters[-num_outputlayer_param:]
    free_grad_param = grad_parameters[:-num_outputlayer_param]
    outputlayer_free_grad_param = filter_free_parameters(outputlayer_grad_param)
    free_grad_param.extend(outputlayer_free_grad_param)
    return free_grad_param

def get_free_vec_parameters_idx(model):
    param = list(model.parameters())
    outputlayer_param = list(model.outputlayer.parameters())
    num_outputlayer_param = len(list(model.outputlayer.parameters()))
    free_param = param[:-num_outputlayer_param]
    count = 0
    idx = None
    if(len(free_param)!=0):
        for i in range(len(free_param)):
            count = count + torch.numel(free_param[i])
        idx = torch.arange(count).long()
    for p in outputlayer_param:
        if(p.size()[0]>1):
            cur_num_free_param = torch.numel(p[:-1,])
        else:
            cur_num_free_param = torch.numel(p)
        total_num_free_param = torch.numel(p)
        if(idx is None):
            idx = torch.arange(count,count+cur_num_free_param).long()
        else:
            idx = torch.cat((idx,torch.arange(count,count+cur_num_free_param).long()),dim=0)
        count = count+total_num_free_param
    if(next(model.parameters()).is_cuda):
        idx = idx.cuda()
    return idx
            
            
def count_free_parameters(model):
    free_param = get_free_parameters(model)
    vec_free_params = vectorize_parameters(free_param)
    num_free_params = vec_free_params.size()[0]
    return num_free_params

def get_AIC(dataSize,model):
    num_param = count_free_parameters(model)
    AIC = (num_param)/dataSize
    return AIC

def get_BIC(dataSize,model):
    num_param = count_free_parameters(model)
    BIC = (num_param*np.log(dataSize))/(2*dataSize)
    return BIC

def get_BC(dataSize,models,loss,coef=2/3):
    BC = np.zeros(len(models))
    AIC = get_regularization(dataSize,models,loss,'AIC')
    AIC_selected = np.argmin(AIC)
    AIC_selected_num_parameters = count_free_parameters(models[AIC_selected])
    for i in range(len(models)):
        num_param = count_free_parameters(models[i])
        if(num_param<=AIC_selected_num_parameters):
            adjustment = 0
            for j in range(1,num_param+1):
                adjustment = adjustment + 1/j
            BC[i] = torch.mean(loss[i])+((dataSize[i]**coef*adjustment)/(2*dataSize[i]))
        else:
            BC[i] = np.inf
    return BC
    
def get_GTIC(dataSize,model,loss_batch):
    likelihood_batch = -loss_batch
    l_vec_free_grad_params=[]
    num_param_outputlayer = model.outputlayer.parameters()
    for j in range(np.int(dataSize)):
        grad_params = torch.autograd.grad(likelihood_batch[j], model.parameters(), create_graph=True, only_inputs=True)
        tmp_free_grad_params = get_free_grad_parameters(grad_params,model)
        vec_free_grad_params = vectorize_parameters(tmp_free_grad_params)
        vec_free_grad_params = vec_free_grad_params.unsqueeze(0)
        l_vec_free_grad_params.append(vec_free_grad_params)       
    free_grad_params = torch.cat(l_vec_free_grad_params,dim=0)
    sum_free_grad_params = torch.sum(free_grad_params,dim=0)
    non_zero_idx = torch.nonzero(sum_free_grad_params[:,0])
    non_zero_idx = non_zero_idx.data
    free_grad_params_T = free_grad_params.transpose(1,2)
    J_batch = torch.matmul(free_grad_params,free_grad_params_T)
    J = torch.sum(J_batch,dim=0)
    J = J[non_zero_idx,non_zero_idx.view(1,-1)]   
    J = J/dataSize
    #print(J)
    H = []
    for j in sum_free_grad_params:
        h = torch.autograd.grad(j, model.parameters(), create_graph=True)
        h = vectorize_parameters(h).view(1,-1)   
        H.append(h)
    H = torch.cat(H,dim=0)
    free_vec_parameters_idx = get_free_vec_parameters_idx(model)
    H = H[:,free_vec_parameters_idx]
    H = H[non_zero_idx,non_zero_idx.view(1,-1)]
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
    return GTIC

def get_Lasso(model,regularization_param):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    Lasso = regularization_param * l1_reg
    return Lasso
    
def get_Ridge(model,regularization_param):
    l2_reg = None
    for W in model.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    Ridge = regularization_param * l2_reg
    return Ridge

def get_ElasticNet(model,regularization_param):
    en_reg = None
    for W in model.parameters():
        if en_reg is None:
            en_reg = regularization_param[0]* W.norm(1) + regularization_param[1] * W.norm(2)
        else:
            en_reg = en_reg + regularization_param[0] * W.norm(1) + regularization_param[1] * W.norm(2)
    ElasticNet = en_reg
    return ElasticNet

def get_GREG(dataSize,model,loss_batch,regularization_param,norm_idx):
    g_reg = None
    if 0 in norm_idx:
        g_reg = get_GTIC(dataSize,model,loss_batch)
        norm_idx = norm_idx[norm_idx!=0]
    for W in model.parameters():
        for i in norm_idx:
            if g_reg is None:
                g_reg = regularization_param[i-1]* W.norm(np.float(i))
            else:
                g_reg = g_reg + regularization_param[i-1]* W.norm(np.float(i)) 
    GREG = g_reg
    return GREG

def get_regularization(dataSizes,models,loss_batches,mode,regularization_param=None):
    REG = np.zeros(len(models))
    if(mode=='BC'):
        REG = get_BC(dataSizes,models,loss_batches)  
        return REG
    for i in range(len(models)):
        print(i)
        if(mode=='Base'):
            REG[i] = torch.mean(loss_batches[i])
        elif(mode=='AIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_AIC(dataSizes[i],models[i])            
        elif(mode=='BIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_BIC(dataSizes[i],models[i])  
        elif(mode=='GTIC'):
            REG[i] = torch.mean(loss_batches[i]) + get_GTIC(dataSizes[i],models[i],loss_batches[i])
        elif(mode=='Lasso'):
            REG[i] = torch.mean(loss_batches[i]) + get_Lasso(models[i],regularization_param[0])
        elif(mode=='Ridge'):
            REG[i] = torch.mean(loss_batches[i]) + get_Ridge(models[i],regularization_param[1])
        elif(mode=='ElasticNet'):
            REG[i] = torch.mean(loss_batches[i]) + get_ElasticNet(models[i],regularization_param[:2])
        elif(mode=='GREG'):
            REG[i] = torch.mean(loss_batches[i]) + get_GREG(dataSizes[i],models[i],loss_batches[i],regularization_param,np.array([0,1,2]))
        else:
            print('mode not supported for model selection')
            exit()
    return REG

def get_GTIC_nofree(dataSize,model,loss_batch):
    try:
        likelihood_batch = -loss_batch
        l_vec_free_grad_params=[]
        for j in range(np.int(dataSize)):
            grad_params = torch.autograd.grad(likelihood_batch[j], model.parameters(), create_graph=True, only_inputs=True)
            vec_free_grad_params = vectorize_parameters(grad_params)
            vec_free_grad_params = vec_free_grad_params.unsqueeze(0)
            l_vec_free_grad_params.append(vec_free_grad_params)       
        free_grad_params = torch.cat(l_vec_free_grad_params,dim=0)
        free_grad_params_T = free_grad_params.transpose(1,2)
        J_batch = torch.matmul(free_grad_params,free_grad_params_T)
        J = torch.sum(J_batch,dim=0)
        J = J/dataSize
        print(J)
        sum_free_grad_params = torch.sum(free_grad_params,dim=0)
        H = []
        for j in sum_free_grad_params:
            h = torch.autograd.grad(j, model.parameters(), create_graph=True)
            h = vectorize_parameters(h).view(1,-1)   
            H.append(h)
        H = torch.cat(H,dim=0)
        V = -H/dataSize
        print(V)
        inv_V = torch.inverse(V)
        VmJ = torch.matmul(inv_V,J)
        tVMJ = torch.trace(VmJ)
        print(tVMJ)
        GTIC = tVMJ/dataSize
    except RuntimeError: 
        GTIC = 0
    return GTIC
    
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
    
    