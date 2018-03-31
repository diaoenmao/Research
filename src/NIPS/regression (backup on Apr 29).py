# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:11:22 2017

@author: jieding and Enmao 
"""
#this files serves to simulate sequential model expansion for regression (with random design)
#design philosophy: create a sequential scenario where data are i.i.d. (X0,y0) potentially nonlinear,
#(1) a fixed candidate set (whenever matrix inverse is not a problem) is prescribed for each t (dataSize), 
#(2) a sliding candidate set is also prescribed to show there is a need to maintain only an active candidate set
#(3) a graph learning based on (1), whose weight plot can illustrate (2) 
#(4) an efficient graph learning that simplifies (3) and achieves (2)   

import numpy as np
from scipy import stats
import matplotlib
import numpy.linalg as lin
import matplotlib.pyplot as plt
from learning import getSeqEdvice
#matplotlib.use("Agg")
import matplotlib.animation as manimation
from mpl_toolkits import axes_grid1
from sklearn import linear_model 
#import IPython
#IPython.start_ipython()

def generate_data(dataSize):
#    np.random.seed(0)
    #X, y = datasets.make_moons(dataSize, noise=0.20)
    X = np.random.randn(dataSize,dataSize)
    #beta = 10.0 / np.power(2, range(0,dataSize))
    beta = 10.0 / np.power(range(1,dataSize+1),2)
    y = X.dot( beta.reshape([dataSize,1]) ) + np.random.randn(dataSize,1)
    return X, y

#candModels is lists storing key-value for each candidate model 
def calculate_loss(candModels, X_test, y_test):
    N = len(candModels)
    loss = np.zeros((N,1))
    for n in range(N):
        var, beta = candModels[n]['var'], candModels[n]['beta']
        if beta is not None:
            loss[n] = np.mean( np.power( y_test - np.concatenate((X_test[:,var],np.ones((X_test.shape[0],1))),axis=1).dot(beta), 2 ), 0 )
        else:
            loss[n] = -np.inf
    return loss

#fit linear model at time t
def linearFit(candModels, Xt,  yt):
    N = len(candModels)
    for n in range(N):
        var = candModels[n]['var']
        if len(var) > len(yt):
            candModels[n]['beta'] = None
        else:
            vals = lin.lstsq(np.concatenate((Xt[:,var],np.ones((Xt.shape[0],1))),axis=1), yt)
            if len(vals[1]) == 0: #sum of residuals is empty 
                candModels[n]['beta'] = None
            else:
                candModels[n]['beta'] = vals[0].reshape([len(var)+1,1])
                candModels[n]['lossEst'] = (2*vals[1][0] + 2 * len(var)) /len(yt) #TIC adjust
    return candModels

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
    
def viewLoss(L_transformed, actSet_start, actSet_end):
    nCandi, T = L_transformed.shape
    optModelIndex = np.argmin(L_transformed, axis=0) #the first occurrence are returned.
    plt.figure(num=1, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.imshow(L_transformed, cmap=plt.cm.Spectral)  #smaller the better
    plt.colorbar()
    
    #plot along the active sets 
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
#    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
#    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,nCandi)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Predictive Loss (in log)')
    plt.show()

def videoLoss(L_transformed, actSet_start, actSet_end):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    nCandi, T = L_transformed.shape
    tmp_L_transformed = np.zeros((nCandi, T))* np.nan
    optModelIndex = np.argmin(L_transformed, axis=0) #the first occurrence are returned.
    #tmp_optModelIndex = np.zeros(len(optModelIndex)))* np.nan
    #fig=plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    fig=plt.figure(facecolor='w', edgecolor='k')
    #plt.imshow(L_transformed, cmap=plt.cm.Spectral)  #smaller the better
    bg=plt.imshow(L_transformed, cmap=plt.cm.Spectral)
    # divider = make_axes_locatable(fig)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    add_colorbar(bg)
    
    #plot along the active sets 
    #plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
    dots=plt.scatter(range(T), optModelIndex, marker='o', color='k', s=10)
#    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
#    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,nCandi)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Predictive Loss (in log)')
    with writer.saving(fig, "Loss.mp4", T):
        for i in range(T):
            tmp_L_transformed[:,i] = L_transformed[:,i]
            bg.set_array(tmp_L_transformed)
            x = range(i+1)
            y = optModelIndex[:i+1]
            offsets = [[x[j],y[j]] for j in range(i+1)]
            dots.set_offsets(offsets)
            writer.grab_frame()
    
def viewSeqWeight(W_hy, L):
    dT, T = W_hy.shape
    plt.figure(num=2, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.imshow(W_hy, cmap='hot')  #smaller the better
    plt.colorbar()
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,dT)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Learning Weight')
    
    #plot along the true best model 
    optModelIndex = np.argmin(L, axis=0)
    plt.scatter(range(T), optModelIndex, marker='o', color='b', s=30)
    plt.show()

def videoSeqWeight(W_hy, L):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    dT, T = W_hy.shape
    tmp_W_hy = np.zeros((dT, T))* np.nan
    optModelIndex = np.argmin(L, axis=0)
    fig=plt.figure(facecolor='w', edgecolor='k') 
    bg=plt.imshow(W_hy, cmap='hot')  #smaller the better
    add_colorbar(bg)
    dots=plt.scatter(range(T), optModelIndex, marker='o', color='b', s=10)
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,dT)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Learning Weight')
    #plot along the true best model 
    with writer.saving(fig, "SeqWeight.mp4", T):
        for i in range(T):
            tmp_W_hy[:,i] = W_hy[:,i]
            bg.set_array(tmp_W_hy)
            x = range(i+1)
            y = optModelIndex[:i+1]
            offsets = [[x[j],y[j]] for j in range(i+1)]
            dots.set_offsets(offsets)
            writer.grab_frame()

def viewSeqLoss(predL_transformed, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(num=3, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    plt.plot(range(t_start-1, T), predL_transformed[t_start-1:T], 'b-', label='Predictor', linewidth=3)  #smaller the better
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()

    
def videoSeqLoss(predL_transformed, L_transformed, t_start):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    dT, T = L_transformed.shape
    fig=plt.figure(facecolor='w', edgecolor='k') 
    #line1=plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    #line2=plt.plot(range(t_start-1, T), predL_transformed[t_start-1:T], 'b-', label='Predictor', linewidth=3)  #smaller the better
    line1,=plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    line2,=plt.plot(range(t_start-1, T), predL_transformed[t_start-1:T], 'b-', label='Predictor', linewidth=3)  #smaller the better    
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10})
    with writer.saving(fig, "SeqLoss.mp4", T):
        for i in range(t_start-1, T):
            line1.set_data(range(t_start-1, i+1), np.min(L_transformed[:,t_start-1:i+1], axis=0))
            line2.set_data(range(t_start-1, i+1), predL_transformed[t_start-1:i+1])
            writer.grab_frame()

def runRegExper():
#%% pre-def the observation up to time T 
    T = 100
    #dT = np.floor(T/2).astype(int) #largest candidate model considered
    dT = 20
    K = 10 #num of active models we maintain in sequential learning
    if dT < K:
        print('error in dT specification!')
    t_start = 11 #starting data size
    nJumpExpe = 1*np.sqrt(T)/K #expected #jump 
    learningParams = {'eta':15*np.sqrt(8*nJumpExpe/T), 'alpha':nJumpExpe/(T), 'G':None} #will use getSeqEdvice()
    actSet, move, thresh = range(K), False, 0.35 #input to learning
    subW = np.zeros((K))
    subW[0] = 1    
    
    #pre-def the benchmark (testdata) to compute the expected loss
    X_test, y_test = generate_data(1000) #testing data for loss computation 
    X, y = generate_data(T) #all the training data 
    
    #pre-def the fixed candidates 
    candModels_fix = []
    for t in range(dT):
        candModels_fix.append({'var': range(t+1), 'beta': None, 'lossEst': None})
    L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    
    #sequential procedure -- compute the loss for all candidates
    for t in range(t_start-1,T): #for each sample size
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = linearFit(candModels_fix, Xt, yt)
        #compute the loss matrix
        L[:,t] = np.squeeze(calculate_loss(candModels_fix, X_test, y_test))
        
    #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT,T))
    seqPredLoss = np.zeros((T))
    loss_ratio =  np.zeros((1,T))    
    for t in range(t_start-1,T): #for each sample size 
        seqPredLoss[t] = np.sum(subW * L[actSet,t]) #should be real loss numerically computed 
        
        #Xt = X[range(t),:]
        #yt = y[range(t),0]
        #candModels_fix = linearFit(candModels_fix, Xt, yt) #already initialized!   
        if t % 10 == 0:
            print("At iteration t = ", t)
        if move:
            actSet = [x+1 for x in actSet]
            if max(actSet) >= dT:
                actSet = range(dT-K, dT)
                move = False
        actSet_start[t], actSet_end[t] = min(actSet), max(actSet)
        #the following expert is assigned its "estimated" loss, which is within sample loss + penalty 
        subE = np.array([ (candModels_fix[i]['lossEst']) for i in actSet]).reshape(K,) #experts
        subW, masterE, move = getSeqEdvice(subE, subW, learningParams, move, thresh, t)
#        print move
        W_hy[actSet,t] = subW   
    
        #newly add
        xxIC = [candModels_fix[n]['lossEst'] for n in range(dT)]
        #print(xxIC)
        n_slect=np.argmax(W_hy[:,t],axis=0)
        #n_slect = xxIC.index(max(xxIC))
        #print(n_slect)
        #print 'hi', n_slect, L[n_slect,t], np.min(L[:,t])
        loss_ratio[0,t] = L[n_slect,t]/np.min(L[:,t])
        
    #summarize results
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    #viewLoss(np.log(L), actSet_start, actSet_end) #viewLoss(L)
    viewSeqWeight(W_hy, L) #print subW
    viewSeqLoss(np.log(seqPredLoss), np.log(L), t_start)
    #viewSeqLoss(seqPredLoss, L, t_start)
    #videoLoss(np.log(L), actSet_start, actSet_end)
    #videoSeqWeight(W_hy, L)
    #videoSeqLoss(np.log(seqPredLoss), np.log(L), t_start)
    plt.figure(num=5, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), loss_ratio[0,t_start-1:T], 'k-', label='Optimum', linewidth=3)  
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%% AR part %%%%%%%%%%%%%%%%%%%%%%%
#dT is added so that we know how many (extra) initial data is needed to start AR fitting 
def generate_AR_data(dataSize, dT): 
#    np.random.seed(0)
    #X, y = datasets.make_moons(dataSize, noise=0.20)
    #beta = 10.0 / np.power(2, range(0,dataSize))
#    L = 2
    L = dT
    y = np.zeros((dataSize+L))
    #true data is from AR
#    beta = np.array([2, 1, -0.8]).reshape([3,1]) #first is a coeff. xn = 2 + xn-1 * 1 + xn-2 * (-0.8)
#    for i in range(dataSize):
#        y[i+2] = beta[0] + beta[1] * y[i] + beta[2] * y[i+1] + np.random.randn(1)
    #true data is from MA 
    e = np.random.randn((dataSize+L))
    for i in range(dataSize+L-1):
        y[i+1] = e[i+1] + 0.8 * e[i]
    
    X = np.ones((dataSize,L+1))
#    X[:,0] = 1
    for k in range(L):
        X[:,k+1] = y[k:(k+dataSize)]
    y = y[L:L+dataSize].reshape([dataSize,1])
    return X, y

#fit linear model at time t
def ARFit(candModels, Xt,  yt):
    N = len(candModels)
    for n in range(N):
        var = candModels[n]['var']
        if len(var) > len(yt):
            candModels[n]['beta'] = None
        else:
            vals = lin.lstsq(Xt[:,var], yt)
            if len(vals[1]) == 0: #sum of residuals is empty 
                candModels[n]['beta'] = None
            else:
                candModels[n]['beta'] = vals[0].reshape([len(var),1])
                candModels[n]['lossEst'] = (vals[1][0] + 2 * len(var)) /len(yt) #TIC adjust
    return candModels

def runARExper():
#%% pre-def the observation up to time T 
    T = 100
    dT = np.floor(T/2).astype(int) #largest candidate model considered 
    K = 3 #num of active models we maintain in sequential learning
    if dT < K:
        print('error in dT specification!')
    t_start = 11 #starting data size
    nJumpExpe = 1*np.sqrt(T)/K #expected #jump 
    learningParams = {'eta':1*np.sqrt(8*nJumpExpe/T), 'alpha':nJumpExpe/T, 'G':None} #will use getSeqEdvice()
    actSet, move, thresh = range(K), False, 0.3 #input to learning
    subW = np.zeros((K))
    subW[0] = 1    
    
    #pre-def the benchmark (testdata) to compute the expected loss
    X_test, y_test = generate_AR_data(1000, dT) #testing data for loss computation 
    X, y = generate_AR_data(T, dT) #all the training data 
    
    #pre-def the fixed candidates 
    candModels_fix = []
    for t in range(dT):
        candModels_fix.append({'var': range(t+1), 'beta': None, 'lossEst': None})
    L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    
    #sequential procedure -- compute the loss for all candidates
    for t in range(t_start-1,T): #for each sample size
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = ARFit(candModels_fix, Xt, yt)
        #compute the loss matrix
        L[:,t] = np.squeeze(calculate_loss(candModels_fix, X_test, y_test))
        
    #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT,T))
    seqPredLoss = np.zeros((T))   
    for t in range(t_start-1,T): #for each sample size 
        seqPredLoss[t] = np.sum(subW * L[actSet,t]) #masterE is wrong! should be real loss numerically computed 
    
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = ARFit(candModels_fix, Xt, yt) #already initialized!   
        if t % 10 == 0:
            print("At iteration t = ", t)
        if move:
            actSet = [x+1 for x in actSet]
            if max(actSet) >= dT:
                actSet = range(dT-K, dT)
                move = False
        actSet_start[t], actSet_end[t] = min(actSet), max(actSet)
        subE = np.array([(candModels_fix[i]['lossEst']) for i in actSet]).reshape(K,) #experts
        subW, masterE, move = getSeqEdvice(subE, subW, learningParams, move, thresh, t)
#        print move
        W_hy[actSet,t] = subW     
    
        
    #summarize results
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    #viewLoss(np.log(L), actSet_start, actSet_end) #viewLoss(L)
    viewSeqWeight(W_hy, L) #print subW
    viewSeqLoss(np.log(seqPredLoss), np.log(L), t_start)
    #viewSeqLoss(seqPredLoss, L, t_start)
    #videoLoss(np.log(L), actSet_start, actSet_end)
    #videoSeqWeight(W_hy, L)
    #videoSeqLoss(np.log(seqPredLoss), np.log(L), t_start)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%% logistic regression part %%%%%%%%%%%%%%%%%%%%%%%
def generate_logistic_data(dataSize, dT): 
    X = np.random.randn(dataSize,dataSize)
    #beta = 10.0 / np.power(2, range(0,dataSize))
    beta = 10.0 / np.power(range(1,dataSize+1),2)
    mu = X.dot( beta.reshape([dataSize,1]) )
    y = np.random.binomial(1, sigmoid(mu), size=None)  
    return X, y

def sigmoid(t): # Define the sigmoid function
    return (1/(1 + np.exp(-t)))       
    

#fit linear model at time t
def LogisticFit(candModels, Xt,  yt):
    N = len(candModels)
    logistic = linear_model.LogisticRegression()
    for n in range(N):
        var = candModels[n]['var']
        if len(var) > len(yt):
            candModels[n]['beta'] = None
        else:
            #vals = lin.lstsq(Xt[:,var], yt)
            logistic.fit(Xt[:,var], yt)
            beta = logistic.coef_.reshape([len(var),1])
            bias = logistic.intercept_
            #print(beta)
            candModels[n]['beta'] = beta
            candModels[n]['bias'] = bias
            Loss,Penalty = getTICLoss(Xt[:,var], yt, beta, bias) #TIC adjusted loss
            candModels[n]['lossEst'] = Loss+Penalty
    return candModels

def getTICLoss(X, y, beta, bias): #get TIC adjusted loss
    mu = X.dot( beta )+ bias
    X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    #print(X)
    loglik = np.sum(mu[y==1]) - np.sum( np.log(1+np.exp(mu)) )
    m = beta.shape[0]+1
    n = y.shape[0]
    J = np.zeros((m,m))
    for i in range(n):
        #J += ( y[i] - np.exp(mu[i])/(1+np.exp(mu[i])) )**2 * X[i,:].T.dot( X[i,:] )
        J += ( y[i] - np.exp(mu[i])/(1+np.exp(mu[i])) )**2 * X[i,:].reshape((m,1)).dot( X[i,:].reshape((1,m)) ) 
    J /= n
    V = np.zeros((m,m))
    for i in range(n): 
        V += np.exp(mu[i]) / ((1+np.exp(mu[i]))**2) * X[i,:].reshape((m,1)).dot( X[i,:].reshape((1,m)) ) 
    V /= n
    #print('J',J)
#    print 'coef', np.exp(mu[i]) / ((1+np.exp(mu[i]))**2)
#    print 'col', X[i,:].reshape((m,1)).dot( X[i,:].reshape((1,m)) )
#    print 'X', X
#    print 'V', V
    pen = np.trace( np.linalg.inv(V).dot(J) ) / n 
    #pen = 1.0*m/n
    #print ('TIC & AIC'), pen, 1.0*m/n
    #print 'loss', -loglik/n 
    #print(-loglik/n)
    #print(pen)
    return -loglik/n,pen 

def calculate_LogisticLoss(candModels, X_test, y_test):
    N = len(candModels)
    loss = np.zeros((N,1))
    for n in range(N):
        var, beta, bias = candModels[n]['var'], candModels[n]['beta'], candModels[n]['bias']
        if beta is not None:
            mu = X_test[:,var].dot( beta ) + bias
            loss[n] = -(np.sum(mu[y_test==1]) - np.sum( np.log(1+np.exp(mu)) ))/y_test.shape[0]
        else:
            loss[n] = -np.inf
    return loss #negative loglik

def runLogisticExper():
#%% pre-def the observation up to time T 
    T = 100
    #dT = np.floor(T/2).astype(int) #largest candidate model considered 
    dT = 10
    K = 3 #num of active models we maintain in sequential learning
    if dT < K:
        print('error in dT specification!')
    t_start = 11 #starting data size
    nJumpExpe = 1*np.sqrt(T)/K #expected #jump 
    learningParams = {'eta':1*np.sqrt(8*nJumpExpe/T), 'alpha':nJumpExpe/T, 'G':None} #will use getSeqEdvice()
    actSet, move, thresh = range(K), False, 0.35 #input to learning
    subW = np.zeros((K))
    subW[0] = 1    
    
    #pre-def the benchmark (testdata) to compute the expected loss
    X_test, y_test = generate_logistic_data(1000, dT) #testing data for loss computation 
    X, y = generate_logistic_data(T, dT) #all the training data 
    
    #pre-def the fixed candidates 
    candModels_fix = []
    for t in range(dT):
        candModels_fix.append({'var': range(t+1), 'beta': None, 'bias': None, 'lossEst': None})
    L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    
    #sequential procedure -- compute the loss for all candidates
    for t in range(t_start-1,T): #for each sample size
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = LogisticFit(candModels_fix, Xt, yt)
        #compute the loss matrix
        L[:,t] = np.squeeze(calculate_LogisticLoss(candModels_fix, X_test, y_test))
        #print(L[:,t])
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    
    #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT,T))
    seqPredLoss = np.zeros((T))  
    loss_ratio =  np.zeros((1,T))
    for t in range(t_start-1,T): #for each sample size 
        seqPredLoss[t] = np.sum(subW * L[actSet,t]) #masterE is wrong! should be real loss numerically computed 
    
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = LogisticFit(candModels_fix, Xt, yt) #already initialized!   
        if t % 10 == 0:
            print("At iteration t = ", t) 
        if move:
            actSet = [x+1 for x in actSet]
            if max(actSet) >= dT:
                actSet = range(dT-K, dT)
                move = False
        actSet_start[t], actSet_end[t] = min(actSet), max(actSet)
        subE = np.array([(candModels_fix[i]['lossEst']) for i in actSet]).reshape(K,)
        subW, masterE, move = getSeqEdvice(subE, subW, learningParams, move, thresh, t)
#        print move
        W_hy[actSet,t] = subW 
        
        #newly add
        xxIC = [candModels_fix[n]['lossEst'] for n in range(dT)]
        #print(xxIC)
        n_slect=np.argmax(W_hy[:,t],axis=0)
        #n_slect = xxIC.index(max(xxIC))
        #print(n_slect)
        #print 'hi', n_slect, L[n_slect,t], np.min(L[:,t])
        loss_ratio[0,t] = L[n_slect,t]/np.min(L[:,t])
    
    #summarize results
    #viewLoss(np.log(L), actSet_start, actSet_end) #viewLoss(L)
    viewSeqWeight(W_hy, L) #print subW
    viewSeqLoss(np.log(seqPredLoss), np.log(L), t_start)
    #viewSeqLoss(seqPredLoss, L, t_start)
    #videoLoss(np.log(L), actSet_start, actSet_end)
    #videoSeqWeight(W_hy, np.log(L))
    #videoSeqLoss(np.log(seqPredLoss), np.log(L), t_start)

    #show efficiency by plottign loss_ratio
    plt.figure(num=5, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), loss_ratio[0,t_start-1:T], 'k-', label='Optimum', linewidth=3)  
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
def main():
# runRegExper()
#    runARExper()
   runLogisticExper()

    
#%%   
if __name__ == "__main__":
    main()    