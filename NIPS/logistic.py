from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import sys
import timeit
import numpy as np
from learning import getSeqEdvice
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
import matplotlib.animation as manimation
from mpl_toolkits import axes_grid1
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib


class LogisticRegression(object):

    def __init__(self, n_data, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_data = n_data
        # self.W = theano.shared(
            # value=np.zeros(
                # (self.dim_in,self.dim_out),
                # dtype=theano.config.floatX
            # ),
            # name='W',
            # borrow=True
        # )
        # self.b = theano.shared(
            # value=np.zeros(
                # (self.dim_out,),
                # dtype=theano.config.floatX
            # ),
            # name='b',
            # borrow=True
        # )
        
    def setParam(self, param):
        beta=np.float32(param[0])
        bias=np.float32(param[1])
        # self.dim_in = beta.shape[0]
        # self.dim_out = beta.shape[1]
        DesignMatrix = np.concatenate((beta,bias.reshape((1,self.dim_out))),axis=0)
        tmp_param = DesignMatrix.reshape(((self.dim_in+1)*self.dim_out,))
        self.param = theano.shared(
            value=tmp_param,
            name='param',
            borrow=True
        )
        #self.param.set_value(tmp_param,borrow=True)
        tmp = self.param.reshape(((self.dim_in+1, self.dim_out)))
        self.W = tmp[:self.dim_in,]
        self.b = tmp[-1,]
        #del beta,bias,DesignMatrix,tmp_param
        
        
    def setData(self, dataX, dataY):
        self.input = theano.shared(
            value=dataX,
            name='input',
            borrow=True
        )
        self.output = theano.shared(
            value=dataY,
            name='output',
            borrow=True
        )
        #self.n_data = dataSize
    
    def setup(self,param,dataX,dataY):
        self.setParam(param)
        self.setData(dataX,dataY)
        self.p_y_given_x = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        
    def negative_log_likelihood_vector(self):
        if(self.dim_out==1):
            tmp = T.log(T.concatenate([(1-self.p_y_given_x),(self.p_y_given_x)],axis=1)[T.arange(self.n_data), self.output])
        else:
            tmp = T.log(self.p_y_given_x)[T.arange(self.n_data), self.output]       
        return -tmp
                
    def negative_log_likelihood_mean(self):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        #print(tmp.eval({self.input : np.random.rand(10, 1),y: np.random.randint(2,size=10)}))
        if(self.dim_out==1):
            tmp = T.log(T.concatenate([(1-self.p_y_given_x),(self.p_y_given_x)],axis=1)[T.arange(self.n_data), self.output])
        else:
            tmp = T.log((self.p_y_given_x)[T.arange(self.n_data), self.output])      
        return -T.mean(tmp)

    # def gradient(self,y):
        # return T.grad(self.negative_log_likelihood_mean(y), self.param)
                
    def TIC_gradient(self):        
        NLLV = self.negative_log_likelihood_vector()
        # tmp = np.zeros((self.dim_in+1,self.dim_in+1))
        # for i in range(self.n_data):
            # tmp = tmp + T.dot(T.grad(NLLV[i],self.param).reshape((-1,1)), T.grad(NLLV[i],self.param).reshape((-1,1)).T)            
        # #print(T.dot(T.grad(t_nll[1],self.param).reshape((-1,1)), T.grad(t_nll[1],self.param).reshape((-1,1)).T).eval({self.input : np.random.rand(10, 1),y: np.random.randint(2,size=10)}))
        # out = tmp/self.n_data
        # #del NLLV,tmp
        
        
        O = T.fmatrix('O')
        o=np.zeros((self.dim_in+1,self.dim_in+1),dtype='float32')
        results, updates = theano.scan(lambda i, tmp: T.dot(T.grad(NLLV[i],self.param).reshape((-1,1)), T.grad(NLLV[i],self.param).reshape((-1,1)).T)+tmp,
                  sequences=[np.arange(self.n_data)],
                  outputs_info=[O])
        result = results[-1]
        compute = theano.function([O], outputs=result)
        out = compute(o)/self.n_data
        return out
        
    def TIC_hessian(self):
        #tmp = T.concatenate([T.log(1-self.p_y_given_x),T.log(self.p_y_given_x)],axis=1)[T.arange(y.shape[0]), y]
        NLLM = self.negative_log_likelihood_mean()
        #print(theano.gradient.hessian(NLLM,self.param).eval({x: np.float32(np.random.randint(2,size=(99,1))), y:np.random.randint(2,size=(99,))}))
        #sum_nll = -T.mean(tmp)
        #print('ccc')
        #print(sum_nll.eval({self.input : dataX, y: dataY}))
        #print(self.param.get_value())
        #print(self.param.eval())
        #print(theano.gradient.hessian(NLLM,self.param))
        H_theta = theano.gradient.hessian(NLLM,self.param)
        #print(H_theta.eval({self.input : dataX, y: dataY}))
        #del NLLM
        return H_theta
        
    def TIC(self):
        #print(dataY.shape)
        #tmp_g = self.gradient(y)
        #tmp_V = self.TIC_hessian(y,dataX,dataY)
        #tmp_V_inv = T.nlinalg.matrix_inverse(self.TIC_hessian(y))
        #tmp_J = self.TIC_gradient(x,y,dataX,dataY)
        #tmp_Dot = T.dot(T.nlinalg.matrix_inverse(self.TIC_hessian(y)),self.TIC_gradient(x,y,dataX,dataY))
        #tmp_overall = T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(self.TIC_hessian(y,dataX,dataY)),self.TIC_gradient(x,y,dataX,dataY)))/self.input.shape[0]
        #print('aaa')
        #print(self.param.get_value())
        #print(tmp_g.eval({x : dataX, y: dataY}))
        #print(tmp_V.eval({x : dataX, y: dataY}))
        #print(tmp_V_inv.eval({x : dataX, y: dataY}))
        #print(tmp_J)
        #print(tmp_Dot.eval({x : dataX, y: dataY}))
        #print(tmp_overall.eval({x : dataX, y: dataY}))
        tmp = T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(self.TIC_hessian()),self.TIC_gradient()))/self.n_data
        return tmp

def sigmoid(t): # Define the sigmoid function
    return (1/(1 + np.exp(-t)))     
    
def generate_logistic_data(dataSize, dT): 
    X = np.random.randn(dataSize,dataSize)
    #beta = 10.0 / np.power(2, range(0,dataSize))
    beta = 10.0 / np.power(range(1,dataSize+1),2)
    mu = X.dot( beta.reshape([dataSize,1]) )
    y = np.random.binomial(1, sigmoid(mu), size=None)  
    return X, y
    
def data2Shared(X,y,borrow=True):
    shared_x = theano.shared(np.asarray(X,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, shared_y
    
def LogisticFit(candModels, Xt, yt):
    N = len(candModels)
    logistic = linear_model.LogisticRegression()
    for n in range(N):
        #print(n)
        var = candModels[n]['var']
        if len(var) > len(yt):
            candModels[n]['beta'] = None
            candModels[n]['bias'] = None
        else:
            #vals = lin.lstsq(Xt[:,var], yt)
            logistic.fit(Xt[:,var], yt)
            beta = logistic.coef_.reshape([len(var),1])
            bias = logistic.intercept_
            #print(bias)
            candModels[n]['beta'] = beta
            candModels[n]['bias'] = bias
            #if(candModels[n]['TICkernel'] is None):
            #x = T.dmatrix('x')  # data, presented as rasterized images
            dataX = Xt[:,var]
            dataY = yt
            #reshape_yt=yt.reshape((-1,1))
            TICKernel = LogisticRegression(n_data=dataX.shape[0], dim_in = beta.shape[0], dim_out = bias.shape[0])
            # TICKernel.setParam([beta,bias])
            # TICKernel.setData(dataX,dataY)            
            # TICKernel.setPYgivenX()
            TICKernel.setup([beta,bias],dataX,dataY)
            loss = TICKernel.negative_log_likelihood_mean()
            penalty = TICKernel.TIC()
            TIC_model = theano.function(
                inputs=[],
                outputs=[loss,penalty],
            )
            Loss,Penalty=TIC_model()
            print(Loss)
            print(Penalty)
            candModels[n]['lossEst'] = Loss+Penalty #TIC adjusted loss
            getTICLoss(dataX, dataY, beta, bias)
            del dataX,dataY, TICKernel,loss,penalty,TIC_model
            #print(candModels[n]['lossEst'])
    del Xt, yt
    return candModels

# def getLoss(x, dataX, dataY, classifier):
    # # x = T.dmatrix('x')  # data, presented as rasterized images
    # # print(x)
    # y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    # loss = classifier.negative_log_likelihood(y)
    # penalty = classifier.TIC(x,y,dataX,dataY)
    # TIC_model = theano.function(
        # inputs=[x,y],
        # outputs=[loss,penalty],
    # )
    # Loss,Penalty = TIC_model(dataX,dataY)
    # print(Loss)
    # print(Penalty)
    # tmp = Loss+Penalty
    # del classifier,x,y,loss,penalty,TIC_model,Loss,Penalty,dataX,dataY
    # return tmp #TIC adjusted loss
            
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
    print(-loglik/n)
    print(pen)
    return -loglik/n + pen 

def calculate_LogisticLoss(candModels, X_test, y_test):
    N = len(candModels)
    loss = np.zeros((N,1))
    for n in range(N):
        var, beta, bias = candModels[n]['var'], candModels[n]['beta'], candModels[n]['bias']
        if beta is not None:
            #print(beta[0],beta[1])
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
    X, Y = generate_logistic_data(T, dT) #all the training data 
    
    #pre-def the fixed candidates 
    candModels_fix = []
    for t in range(dT):
        candModels_fix.append({'var': range(t+1), 'beta': None, 'bias': None, 'lossEst': None})
    L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    
    #dim_out = 1
    #x = theano.tensor.fmatrix('x')
    #y = theano.tensor.ivector('y')
    #TICKernel = LogisticRegression(input=x, output=y, n_data=X.shape[0], dim_in=X.shape[1], dim_out=dim_out)
    
    #sequential procedure -- compute the loss for all candidates
    for t in range(t_start-1,T): #for each sample size
        print(t)
        Xt = np.float32(X[range(t),:])
        yt = np.int32(Y[range(t),0])
        tmp = LogisticFit(candModels_fix, Xt, yt)
        #compute the loss matrix
        L[:,t] = np.squeeze(calculate_LogisticLoss(tmp, X_test, y_test))
        #del Xt,yt,tmp
    L[L==-np.inf] = np.max(L) #the none are set to be the max loss
    
    #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT,T))
    seqPredLoss = np.zeros((T))  
    loss_ratio =  np.zeros((1,T))
    for t in range(t_start-1,T): #for each sample size 
        seqPredLoss[t] = np.sum(subW * L[actSet,t]) #masterE is wrong! should be real loss numerically computed 

        if t % 10 == 0:
            print("At iteration t = ", t) 
        if move:
            actSet = [x+1 for x in actSet]
            if max(actSet) >= dT:
                actSet = range(dT-K, dT)
                move = False
        actSet_start[t], actSet_end[t] = min(actSet), max(actSet)
        subE = np.array([L[i,t] for i in actSet]).reshape(K,) #experts
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

    #print(test_set_y[500:1600].eval())
    
if __name__ == '__main__':
    sys.setrecursionlimit(3000)
    #theano.config.exception_verbosity='high'
    theano.config.allow_input_downcast=True
    runLogisticExper()