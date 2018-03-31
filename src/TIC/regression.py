# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:11:22 2017

@author: jieding
"""
#this files serves to simulate sequential model expansion for regression (with random design)
#design philosophy: create a sequential scenario where data are i.i.d. (X0,y0) potentially nonlinear,
#(1) a fixed candidate set (whenever matrix inverse is not a problem) is prescribed for each t (dataSize), 
#(2) a sliding candidate set is also prescribed to show there is a need to maintain only an active candidate set
#(3) a graph learning based on (1), whose weight plot can illustrate (2) 
#(4) an efficient graph learning that simplifies (3) and achieves (2)   

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from learning import getSeqEdvice
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
            loss[n] = np.mean( np.power( y_test - X_test[:,var].dot(beta), 2 ), 0 )
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
            vals = lin.lstsq(Xt[:,var], yt)
            if len(vals[1]) == 0: #sum of residuals is empty 
                candModels[n]['beta'] = None
            else:
                candModels[n]['beta'] = vals[0].reshape([len(var),1])
                candModels[n]['lossEst'] = (vals[1][0] + 2 * len(var)) /len(yt) #TIC adjust
    return candModels

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
    
def main():
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
    for t in range(t_start-1,T): #for each sample size 
        Xt = X[range(t),:]
        yt = y[range(t),0]
        candModels_fix = linearFit(candModels_fix, Xt, yt) #already initialized!   
        if t % 10 == 0:
            print("At iteration t = ", t)
        if move:
            actSet = [x+1 for x in actSet]
            if max(actSet) >= dT:
                actSet = range(dT-K, dT)
                move = False
        actSet_start[t], actSet_end[t] = min(actSet), max(actSet)
        subE = np.array([candModels_fix[i]['lossEst'] for i in actSet]).reshape(K,) #experts
        subW, masterE, move = getSeqEdvice(subE, subW, learningParams, move, thresh, t)
#        print move
        W_hy[actSet,t] = subW 
        seqPredLoss[t] = np.sum(subW * L[actSet,t]) #masterE is wrong! should be real loss numerically computed 
    
        
    #summarize results
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    viewLoss(np.log(L), actSet_start, actSet_end) #viewLoss(L)
    viewSeqWeight(W_hy, L) #print subW
    #viewSeqLoss(np.log(seqPredLoss), np.log(L), t_start)
    viewSeqLoss(seqPredLoss, L, t_start)
    
#%%   
if __name__ == "__main__":
    main()    