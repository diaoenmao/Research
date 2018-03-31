dataName = 'MNIST original'
data_dir = './data/'



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
