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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import copy
import pickle
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
                candModels[n]['lossEst'] = (vals[1][0] + len(var)) /len(yt) #TIC adjust
    return candModels

def runRegExper():
#%% pre-def the observation up to time T 
    T = 200
    #dT = np.floor(T/2).astype(int) #largest candidate model considered
    dT = np.floor(np.sqrt(T)).astype(int)
    K = 3 #num of active models we maintain in sequential learning
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
    plt.figure(num=5, figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
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
    #dT = np.floor(T/2).astype(int) #largest candidate model considered 
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
    
        # Xt = X[range(t),:]
        # yt = y[range(t),0]
        # candModels_fix = ARFit(candModels_fix, Xt, yt) #already initialized!   
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
    X = np.random.randn(dataSize,100)
    #beta = 10.0 / np.power(2, range(0,dataSize))
    beta = 10.0 / np.power(range(1,100+1),1.5)
    mu = X.dot( beta.reshape([100,1]) )
    y = np.random.binomial(1, sigmoid(mu), size=None)  
    return X, y

def crossvalition_p(X,y,p):
    dataSize = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-p), random_state=40)
    return (X_train, X_test, y_train, y_test)
    
def crossvalition_10fold(X,y):
    tenfold = KFold(n_splits=10)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in tenfold.split(X, y):
        X_train.append(X[train_index,:])
        y_train.append(y[train_index])
        X_test.append(X[test_index,:])
        y_test.append(y[test_index])    
    return (X_train, X_test, y_train, y_test)
    
def crossvalition_loo(X,y):    
    loo = KFold(n_splits=X.shape[0])
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in loo.split(X, y):
        X_train.append(X[train_index,:])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    #print(X_train.shape)
    #print(X_test[0].shape)
    #print(X.shape)
    return (X_train, X_test, y_train, y_test)

def sigmoid(t): # Define the sigmoid function
    return (1/(1 + np.exp(-t)))       
    

#fit linear model at time t
def LogisticFit(candModels, Xt,  yt):
    N = len(candModels)
    kmax = 0
    Loss_AIC = np.inf
    logistic = linear_model.LogisticRegression(warm_start=True)
    timing = {'AIC': np.zeros((1,N)), 'BIC': np.zeros((1,N)), 'DIC_1': np.zeros((1,N)), 'DIC_2': np.zeros((1,N)), 'TIC': np.zeros((1,N)), \
        'Holdout': np.zeros((1,N)), 'CV_10fold': np.zeros((1,N)), 'CV_loo': np.zeros((1,N))}
    for n in range(N):
        print(n)
        var = candModels[n]['var']
        if Xt.shape[1] > len(yt[var]):
            candModels[n]['beta'] = None
            candModels[n]['bias'] = None
        else:
            if(Xt.shape[1]<np.floor(np.sqrt(Xt.shape[0])).astype(np.int) or Xt.shape[1] < 3):
            #if(Xt.shape[1]<np.floor(np.sqrt(Xt.shape[0])).astype(np.int) or Xt.shape[1] < 3 or True):  
                logistic_cv = linear_model.LogisticRegression()
                #Holdout
                start=time.time()
                #print(var)
                #print(Xt[var,:].shape)
                #print(yt[var].shape)
                X_train, X_test, y_train, y_test=crossvalition_p(Xt[var,:], yt[var], 0.7)
                logistic_cv.fit(X_train, y_train)
                beta = logistic_cv.coef_.reshape([Xt.shape[1],1])
                bias = logistic_cv.intercept_
                Loss = getLoss(X_test, y_test, beta, bias)
                candModels[n]['Holdout'] = Loss
                end=time.time()
                timing['Holdout'][:,n] = timing['Holdout'][:,n]+end-start
                #CV_10fold
                start=time.time()            
                X_train, X_test, y_train, y_test=crossvalition_10fold(Xt[var,:], yt[var])
                Loss=np.zeros(len(X_train))
                for i in range(len(X_train)):
                    logistic_cv.fit(X_train[i], y_train[i])
                    beta = logistic_cv.coef_.reshape([Xt.shape[1],1])
                    bias = logistic_cv.intercept_
                    Loss[i]=getLoss(X_test[i], y_test[i], beta, bias)
                candModels[n]['CV_10fold'] = np.mean(Loss)
                end=time.time()
                timing['CV_10fold'][:,n] = timing['CV_10fold'][:,n]+end-start
                #CV_loo        
                start=time.time()             
                X_train, X_test, y_train, y_test=crossvalition_loo(Xt[var,:], yt[var])
                Loss=np.zeros(len(X_train))
                for i in range(len(X_train)):
                    logistic_cv.fit(X_train[i], y_train[i])
                    beta = logistic_cv.coef_.reshape([Xt.shape[1],1])
                    bias = logistic_cv.intercept_
                    Loss[i] = getLoss(X_test[i], y_test[i], beta, bias)
                candModels[n]['CV_loo'] = np.mean(Loss)            
                end=time.time()
                timing['CV_loo'][:,n] = timing['CV_loo'][:,n]+end-start
                
            start=time.time()             
            logistic.fit(Xt[var,:], yt[var])
            beta = logistic.coef_.reshape([Xt.shape[1],1])
            bias = logistic.intercept_
            Loss = getLoss(Xt[var,:], yt[var], beta, bias)
            candModels[n]['beta'] = beta
            candModels[n]['bias'] = bias
            dataSize = Xt[var,:].shape[0]
            dim = (beta.shape[0]+1)*beta.shape[1]
            end=time.time()
            time_loss = end-start
            if(Xt.shape[1]<np.floor(np.sqrt(Xt.shape[0])).astype(np.int) or Xt.shape[1] < 3):
            #if(Xt.shape[1]<np.floor(np.sqrt(Xt.shape[0])).astype(np.int) or Xt.shape[1] < 3 or True):             
                start_AIC=time.time()
                AIC = getAIC(dataSize, dim)
                candModels[n]['AIC'] = Loss+AIC
                end_AIC=time.time()
                timing['AIC'][:,n] = timing['AIC'][:,n]+time_loss+end_AIC-start_AIC
                
                start_BIC=time.time()
                BIC = getBIC(dataSize, dim)
                candModels[n]['BIC'] = Loss+BIC
                end_BIC=time.time()
                timing['BIC'][:,n] = timing['BIC'][:,n]+time_loss+end_BIC-start_BIC

                start_DIC_2=time.time()            
                DIC_2 = getDIC_2(dataSize, dim, dataSize**(1/2))
                candModels[n]['DIC_2'] = Loss+DIC_2
                end_DIC_2=time.time()
                timing['DIC_2'][:,n] = timing['DIC_2'][:,n]+time_loss+end_DIC_2-start_DIC_2

                start_TIC=time.time()            
                TIC = getTIC(Xt[var,:], yt[var], beta, bias) 
                print(Loss)
                print(TIC)
                candModels[n]['TIC'] = Loss+TIC
                end_TIC=time.time()
                timing['TIC'][:,n] = timing['TIC'][:,n]+time_loss+end_TIC-start_TIC
                
                # start_DIC_1=time.time() 
                # if(candModels[n]['AIC']<Loss_AIC):
                    # Loss_AIC = candModels[n]['AIC']
                    # kmax = n
                # end_DIC_1=time.time()
                # timing['DIC_1'][n] = timing['DIC_1'][n]+time_loss+end_DIC_1-start_DIC_1

    # start_DIC_1=time.time()
    # for n in range(N):
        # if(n<kmax):
            # var = candModels[n]['var']
            # if Xt.shape[1] <= len(yt[var]):
                # beta = candModels[n]['beta']
                # bias = candModels[n]['bias']
                # dataSize = Xt[var,:].shape[0]
                # dim = (beta.shape[0]+1)*beta.shape[1]
                # DIC_1 = getDIC_1(dataSize,dim,dataSize**(1/3))
                # candModels[n]['DIC_1'] = Loss+DIC_1
        # else:
            # candModels[n]['DIC_1'] = 0
    # end_DIC_1=time.time()
    # timing['DIC_1'] = timing['DIC_1']+end_DIC_1-start_DIC_1
    return (candModels, timing)

def getLoss(X, y, beta, bias):
    mu = X.dot( beta )+ bias
    #X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    #print(X)
    loglik = np.sum(mu[y==1]) - np.sum( np.log(1+np.exp(mu)) )
    n = y.shape[0]
    return -loglik/n
    
def getTIC(X, y, beta, bias): #get TIC adjusted loss
    mu = X.dot( beta )+ bias
    X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
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
    return pen
    
def getAIC(n,k):
    return k/n

def getBIC(n,k):
    return (k*np.log(n))/(2*n)

def getDIC_1(n,k_AIC,Mn):
    adjust = 0
    for i in range(k_AIC):
        adjust = adjust + 1/(i+1)
    return Mn/n*adjust
    
def getDIC_2(n,k,kmax):
    adjust = 0
    for i in range(k):
        adjust = adjust + 1/(i+1)
    return kmax/n*adjust
    
def calculate_LogisticLoss(candModels, X_test, y_test):
    N = len(candModels)
    loss = np.zeros((N,1))
    for n in range(N):
        var, beta, bias = candModels[n]['var'], candModels[n]['beta'], candModels[n]['bias']
        if beta is not None:
            #print(X_test[var,:beta.shape[0]].shape)
            mu = X_test[var,:beta.shape[0]].dot( beta ) + bias
            #print(y_test[var,:].shape)
            loss[n] = -(np.sum(mu[y_test[var,:]==1]) - np.sum( np.log(1+np.exp(mu))))/y_test[var,:].shape[0]
        else:
            loss[n] = -np.inf
    return loss #negative loglik

def runLogisticExper():
#%% pre-def the observation up to time T 
    T = 101
    dT = 50 #largest candidate model considered 
    #dT = np.sqrt(T).astype(int)
    K = 3 #num of active models we maintain in sequential learning
    if dT < K:
        print('error in dT specification!')
    t_start = 11 #starting data size
    
    nJumpExpe = 1*np.sqrt(T)/K #expected #jump 
    learningParams = {'eta':1*np.sqrt(8*nJumpExpe/T), 'alpha':1*nJumpExpe/T, 'G':None} #will use getSeqEdvice()
    #print(learningParams['alpha'])
    actSet, move, thresh = range(K), False, 1/K #input to learning
    subW = np.zeros((K))
    subW[0] = 1    
    N=10000
    #pre-def the benchmark (testdata) to compute the expected loss
    X_test, y_test = generate_logistic_data(N, dT) #testing data for loss computation 
    X, y = generate_logistic_data(T, dT) #all the training data 
    #pre-def the fixed candidates 
    candModels_fix = []
    for t in range(t_start-1,T):
        candModels_fix.append({'var': range(t+1), 'beta': None, 'bias': None, 'AIC': None, 'BIC': None, 'DIC_1': None, 'DIC_2': None, 'TIC': None, \
                'Holdout': None, 'CV_10fold': None, 'CV_loo': None})
    L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    candModels_time = []
    timing_image = {'AIC': np.array([]), 'BIC': np.array([]), 'DIC_1': np.array([]), 'DIC_2': np.array([]), 'TIC': np.array([]), \
                'Holdout': np.array([]), 'CV_10fold': np.array([]), 'CV_loo': np.array([])}
    #sequential procedure -- compute the loss for all candidates
    # timing = {'AIC': np.zeros(T), 'BIC': np.zeros(T), 'DIC_1': np.zeros(T), 'DIC_2': np.zeros(T), 'TIC': np.zeros(T), \
        # 'Holdout': np.zeros(T), 'CV_10fold': np.zeros(T), 'CV_loo': np.zeros(T)}
    for t in range(dT): #for each sample size
        print(t)
        Xt = X[:,range(t+1)]
        yt = y[:,0]
        candModels_fix,timing = LogisticFit(candModels_fix, Xt, yt)
        for k in timing_image.keys():
            #print(timing[k].shape)
            timing_image[k] = np.vstack([timing_image[k],timing[k]]) if timing_image[k].size else timing[k]

        #compute the loss matrix
        L[t,t_start-1:] = np.squeeze(calculate_LogisticLoss(candModels_fix, X_test, y_test))
        candModels_time.append(copy.deepcopy(candModels_fix))
        #print(candModels_time[t-t_start+1][4]['TIC'])
    for k in timing_image.keys():
        timing_image[k] = np.sum(timing_image[k],axis=0)
        timing_image[k] = np.cumsum(timing_image[k])
        
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    
    #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT,T))
    # seqPredLoss = np.zeros((T))  
    # loss_ratio =  np.zeros((1,T))
    
    
    output_filename = './logistic/output.pickle'
    #mode = ['AIC','BIC','DIC_1','DIC_2','Holdout','CV_10fold','CV_loo','TIC']
    #mode = ['AIC','BIC','DIC_1','DIC_2','TIC','Holdout','CV_10fold','CV_loo']
    mode = ['TIC','Holdout','CV_10fold','CV_loo']
    #mode = ['TIC','AIC','BIC','DIC_1']
    #mode = ['TIC','AIC','BIC']
    candModels_Sequential = []
    for i in range(len(mode)):
        candModels_Sequential.append({'learningParams': learningParams, 'actSet': actSet, 'actSet_start': actSet_start,'actSet_end': actSet_end, 'move': move, 'nummove': 0, 'thresh': thresh, \
            'subW':subW, 'W_hy':W_hy, 'seqPredLoss': np.zeros((T)), 'loss_ratio': np.zeros((T)), 'batch_opt_model': np.zeros((T),dtype=np.int), 'batch_opt_loss': np.zeros((T))+np.inf, 'batch_loss_ratio': np.zeros((T))})   
    for t in range(t_start-1,T): #for each sample size
        cur_dT = np.floor(np.sqrt(t)).astype(np.int)
        #cur_dT = dT
        #print(t)
        #print(candModels_time[t-t_start+1][4]['TIC']) 
        for m in range(len(mode)):
            candModels_Sequential[m]['seqPredLoss'][t] = np.sum(candModels_Sequential[m]['subW'] * L[candModels_Sequential[m]['actSet'],t]) #masterE is wrong! should be real loss numerically computed 
            if t % 10 == 0:
                print("At iteration t = ", t) 
            if candModels_Sequential[m]['move']:
                candModels_Sequential[m]['actSet'] = [(x+1) for x in candModels_Sequential[m]['actSet']]
                candModels_Sequential[m]['nummove'] = candModels_Sequential[m]['nummove'] + 1
                movedsample = t/candModels_Sequential[m]['nummove']
                tmp_nJumpExpe = movedsample/K
                candModels_Sequential[m]['learningParams']['alpha'] = 1*tmp_nJumpExpe/T
                candModels_Sequential[m]['learningParams']['eta'] = 1*np.sqrt(8*tmp_nJumpExpe/T)
                #print(candModels_Sequential[m]['learningParams']['alpha'])
                if max(candModels_Sequential[m]['actSet']) >= dT:
                    candModels_Sequential[m]['actSet'] = range(dT-K, dT)
                    candModels_Sequential[m]['move'] = False               
            candModels_Sequential[m]['actSet_start'][t], candModels_Sequential[m]['actSet_end'][t] = min(candModels_Sequential[m]['actSet']), max(candModels_Sequential[m]['actSet'])
            subE = np.array([(candModels_time[j][t-t_start+1][mode[m]]) for j in candModels_Sequential[m]['actSet']]).reshape(K,)
            candModels_Sequential[m]['subW'], masterE, candModels_Sequential[m]['move'] = getSeqEdvice(subE, candModels_Sequential[m]['subW'], candModels_Sequential[m]['learningParams'], \
                candModels_Sequential[m]['move'], candModels_Sequential[m]['thresh'], t)
            candModels_Sequential[m]['W_hy'][candModels_Sequential[m]['actSet'],t] = candModels_Sequential[m]['subW'] 
            weight=candModels_Sequential[m]['subW']
            candModels_Sequential[m]['loss_ratio'][t] = np.sum(L[candModels_Sequential[m]['actSet'],t]*weight)/np.min(L[:,t])
            #candModels_Sequential[m]['loss_ratio'][t] = L[np.argmax(candModels_Sequential[m]['W_hy'][:,t],axis=0),t]/np.min(L[:,t])
            #print(mode[m])           
            for l in range(cur_dT):
                #print(candModels_time[t-t_start+1][l][mode[m]])
                if((candModels_time[l][t-t_start+1][mode[m]] is not None) and (candModels_time[l][t-t_start+1][mode[m]]<candModels_Sequential[m]['batch_opt_loss'][t])):
                    candModels_Sequential[m]['batch_opt_model'][t] = l
                    candModels_Sequential[m]['batch_opt_loss'][t] = candModels_time[l][t-t_start+1][mode[m]]
            #print(candModels_Sequential[m]['batch_opt_loss'][t])
            candModels_Sequential[m]['batch_loss_ratio'][t] = L[candModels_Sequential[m]['batch_opt_model'][t],t]/np.min(L[:,t])
    
    #summarize results
    viewLoss(np.log(L),candModels_Sequential[mode.index('TIC')]['actSet_start'],candModels_Sequential[mode.index('TIC')]['actSet_end'])
    viewSeqWeight(candModels_Sequential[mode.index('TIC')]['W_hy'], L) #print subW
    #viewSeqLoss_all(mode, candModels_Sequential, L, t_start)
    #viewBatchLoss_all(mode, candModels_Sequential, L, t_start)
    #viewLossRatio_all(mode, candModels_Sequential, t_start, T)
    #viewBatchLossRatio_all(mode, candModels_Sequential, t_start, T)
    #viewSeqBatchLoss(mode, candModels_Sequential, L, t_start)
    viewSeqBatchLossRatio(mode,candModels_Sequential, t_start, T)
    viewTiming(mode,timing_image, t_start, T)
    
    with open(output_filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([candModels_Sequential,candModels_time, L, mode, t_start, T, timing_image], f)
        
    #show efficiency by plottign loss_ratio
    #viewLossRatio(loss_ratio, t_start)
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
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    bg=plt.imshow(L_transformed, cmap=plt.cm.Spectral)  #smaller the better
    #plt.colorbar()
    add_colorbar(bg)
    #plot along the active sets 
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=5)
#    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
#    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0,T-1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    plt.ylim(0,nCandi-1)
    plt.ylabel('Model Complexity', fontsize=10, color='black')
    plt.title('Predictive Loss (in log)', fontsize=10)
    #plt.tight_layout()
    plt.savefig('./logistic/loss.png', bbox_inches='tight')
    plt.show()

def videoLoss(L_transformed, actSet_start, actSet_end):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    nCandi, T = L_transformed.shape
    tmp_L_transformed = np.zeros((nCandi, T))* np.nan
    optModelIndex = np.argmin(L_transformed, axis=0) #the first occurrence are returned.
    #tmp_optModelIndex = np.zeros(len(optModelIndex)))* np.nan
    #fig=plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
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
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k') 
    bg=plt.imshow(W_hy, cmap='hot')  #smaller the better
    #plt.colorbar()
    add_colorbar(bg)
    plt.xlim(0,T-1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    plt.ylim(0,dT-1)
    plt.ylabel('Model Complexity', fontsize=10, color='black')
    plt.title('Learning Weight', fontsize=10)
    
    #plot along the true best model 
    optModelIndex = np.argmin(L, axis=0)
    plt.scatter(range(T), optModelIndex, marker='o', color='b', s=5)
    #plt.tight_layout()
    plt.savefig('./logistic/weight.png', bbox_inches='tight')
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
    dots=plt.scatter(range(T), optModelIndex, marker='o', color='b', s=5)
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
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    plt.plot(range(t_start-1, T), predL_transformed[t_start-1:T], 'b-', label='Predictor', linewidth=3)  #smaller the better
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()

def viewSeqLoss_all(mode, candModels_Sequential, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start-1, T), candModels_Sequential[i]['seqPredLoss'][t_start-1:T], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])  #smaller the better    
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()

def viewBatchLoss_all(mode, candModels_Sequential, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum', linewidth=3)  #smaller the better
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        rows = candModels_Sequential[i]['batch_opt_model'][t_start-1:T].tolist()
        cols = range(t_start-1,T)
        #print(L_transformed[rows,cols])
        plt.plot(range(t_start-1, T), L_transformed[rows,cols], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])  #smaller the better    
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylabel('Batch Loss', fontsize=14, color='black')
    plt.title('Batch Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()
    
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')  
    optModelIndex = np.argmin(L_transformed, axis=0)
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    markers = ['s','*','^','D']
    for i in range(len(mode)):
        plt.scatter(range(t_start-1, T), candModels_Sequential[i]['batch_opt_model'][t_start-1:T], marker=markers[i], color=colors[i], label=mode[i], s=30)
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,dT)
    plt.ylabel('Batch Model', fontsize=14, color='black')
    plt.title('Batch Model of the optimal model and our predictor at each time')
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

def viewSeqBatchLossRatio(mode,candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):  
        plt.plot(range(t_start-1, T),candModels_Sequential[i]['batch_loss_ratio'][t_start-1:T], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.plot(range(t_start-1, T),candModels_Sequential[0]['loss_ratio'][t_start-1:T], color='k', label='TIC_Sequential', linewidth=1, linestyle='--')
    plt.xlim(t_start-1,T-1)
    plt.xlabel('Data Size', fontsize=10, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss Ratio', fontsize=10, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    #plt.tight_layout()
    plt.grid()
    plt.savefig('./logistic/lossratio.png', bbox_inches='tight')
    plt.show()
    
def viewLossRatio(loss_ratio, t_start, T):
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k') 
    plt.plot(range(t_start-1, T), loss_ratio[t_start-1:T], 'k-', label='Optimum', linewidth=3)  
    plt.show()
    
def viewLossRatio_all(mode,candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):  
        plt.plot(range(t_start-1, T), candModels_Sequential[i]['loss_ratio'][t_start-1:T], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Loss Ratio', fontsize=14, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()

def viewBatchLossRatio_all(mode,candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['|', '--', '-.', ':']
    for i in range(len(mode)):  
        plt.plot(range(t_start-1, T), candModels_Sequential[i]['batch_loss_ratio'][t_start-1:T], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim(t_start-1,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Batch Loss Ratio', fontsize=14, color='black')
    plt.title('Batch Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10}) 
    plt.show()
    
def viewTiming(mode,timing_image, t_start,T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):  
        plt.plot(range(t_start-1, T), timing_image[mode[i]], color=colors[i], label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim(t_start-1,T-1)
    plt.xlabel('Data Size', fontsize=10, color='black')
#    plt.ylim(0,dT)
    plt.ylabel('Elapsed Time', fontsize=10, color='black')
    plt.title('Elapsed Time of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size':10})
    #plt.tight_layout()
    plt.grid()
    plt.savefig('./logistic/timing.png', bbox_inches='tight')
    plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
def main():
# runRegExper()
#    runARExper()
   runLogisticExper()

    
#%%   
if __name__ == "__main__":
    main()