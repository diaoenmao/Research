__author__ = 'Jie'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy.linalg as lin

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.00 #0.01  # regularization strength


def generate_data(dataSize):
#    np.random.seed(0)
    X, y = datasets.make_moons(dataSize, noise=0.20)
    return X, y


def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Logistic Regression")
    plt.show()


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    dataSize = len(X)  # training set size
    W1, b1, W2, b2, actFun = model['W1'], model['b1'], model['W2'], model['b2'], model['actFun']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = act(z1, actFun)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(dataSize), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / dataSize * data_loss


def predict(model, x):
    W1, b1, W2, b2, actFun = model['W1'], model['b1'], model['W2'], model['b2'], model['actFun']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = act(z1, actFun)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes, actFun, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    dataSize = len(X)
#    np.random.seed(0) #TAKE CAUTION OF THIS RANDOMNESS
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = act(z1, actFun)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
#        delta3 = probs
#        delta3[range(dataSize), y] -= 1
#        dW2 = (a1.T).dot(delta3)
#        db2 = np.sum(delta3, axis=0, keepdims=True)
#        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
#        dW1 = (X.T).dot(delta2)
#        db1 = np.sum(delta2, axis=0)

        #Below is my customization, where bias b is absorbed into weight W
        delta3 = probs
        delta3[range(dataSize), y] -= 1
        a1_aug = np.append (a1, np.ones((dataSize,1)), 1)
        dW2_aug = (a1_aug.T).dot(delta3)
        dW2 = dW2_aug[0:-1,:]
        db2 = dW2_aug[-1,:]
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        X_aug =np.append (X, np.ones((dataSize,1)), 1)
        dW1_aug = (X_aug.T).dot(delta2)
        dW1 = dW1_aug[0:-1,:]
        db1 = dW1_aug[-1,:]

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'actFun': actFun}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model

#single observation: X is of size 1xnn_input_dim, y is 1xnn_out_dim
#output deriv as col, which is a (9+8)17 * 1 vector for 3 hidden nodes
def getDerivative(model, X0, y0):
    W1, b1, W2, b2, actFun = model['W1'], model['b1'], model['W2'], model['b2'], model['actFun']
    [nn_input_dim, nn_hdim] = W1.shape
    nn_output_dim = W2.shape[1]
    X0 = X0.reshape([1,nn_input_dim])
    
    # Forward propagation
    z1 = X0.dot(W1) + b1
    a1 = act(z1, actFun)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Backpropagation, where bias b is absorbed into weight W
    delta3 = probs
    delta3[:,y0] -= 1
    a1_aug = np.append(a1, np.ones((1,1)), 1)
    dW2_aug = (a1_aug.T).dot(delta3)
    delta2 = delta3.dot(W2.T) * derivActFun(z1, actFun, 1)
    X_aug =np.append(X0, np.ones((1,1)), 1)
    dW1_aug = (X_aug.T).dot(delta2)
    #dW2_aug: [i,j]th element is the weight from node i to j in layer 2, with the last row indicating the biases of all receivers
    #dW1_aug: similar
    deriv = np.append(dW2_aug.reshape([(nn_hdim+1)*nn_output_dim, 1]), dW1_aug.reshape([(nn_input_dim+1)*nn_hdim, 1]), 0)
    return deriv #the two cols do not sum to zero 

#single observation: X is of size 1xnn_input_dim, y is 1xnn_out_dim
#output (9+8)17*17 matrix for 3 hidden nodes
#computing based on Bishop's paper and Peter's note
#Recall from Peter's note that par L / par z2[i] = probs[i] - y[i]
#then the desired H_i = par2 L / par2 z2 in Bishop Eq19 becomes
#par (probs[i] - y[i]) / par z2[i] = par probs[i] / par z2[i] = probs[i] (1-probs[i])
def get2ndDerivative(model, X0, y0):
    W1, b1, W2, b2, actFun = model['W1'], model['b1'], model['W2'], model['b2'], model['actFun']
    [nn_input_dim, nn_hdim] = W1.shape
    nn_output_dim = W2.shape[1]
    X0 = X0.reshape([1,nn_input_dim])
    
    # Forward propagation
    z1 = X0.dot(W1) + b1
    a1 = act(z1, actFun)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    #useful quantities 
    a1_aug = np.append(a1, np.ones((1,1)), 1) #augmentation that absorbes the biase terms
    X_aug =np.append(X0, np.ones((1,1)), 1)
    f1 = derivActFun(z1, actFun, 1)  #f'
    f2 = derivActFun(z1, actFun, 2)  #f'' 
    H = probs * (1-probs)
    sigma = probs - y0
    
    nW_layer2 = (nn_hdim+1)*nn_output_dim
    nW_layer1 = (nn_input_dim+1)*nn_hdim
    Hass = np.zeros((nW_layer2+nW_layer1, nW_layer2+nW_layer1))
    
    #compute for the block corres. layer 2
    for lm in range(0, nW_layer2):
        #map ml to [m,l]th location in W2-matrix. corrs. to derivative's reshape order (row first)
        l = lm / nn_output_dim
        m = lm % nn_output_dim
        for lm_ in range(0, nW_layer2):
            l_ = lm_ / nn_output_dim
            m_ = lm_ % nn_output_dim
            Hass[lm, lm_] = a1_aug[0,l] * a1_aug[0,l_] * H[0,m] * (m==m_)
     
    #compute for the block corres. layer 1
    for kl in range(0, nW_layer1):
        k = kl / nn_hdim
        l = kl % nn_hdim
        for kl_ in range(0, nW_layer1):
            k_ = kl_ / nn_hdim
            l_ = kl_ % nn_hdim
            temp1 = f2[0,l] * (l==l_) * np.sum(W2[l_,:] * sigma, axis=1)
            temp2 = f1[0,l] * f1[0,l_] * np.sum(W2[l,:] * W2[l_,:] * H, axis=1)
            Hass[nW_layer2+kl, nW_layer2+kl_] = X_aug[0,k] * X_aug[0,k_] * (temp1+temp2)
        
    #compute for the block corres. layer 1 cross layer 2
    for kl in range(0, nW_layer1):
        k = kl / nn_hdim
        l = kl % nn_hdim
        for l_m in range(0, nW_layer2):
            l_ = l_m / nn_output_dim
            m = l_m % nn_output_dim
            Hass[nW_layer2+kl, l_m] = X_aug[0,k] * f1[0,l] * (sigma[0,m]*(l==l_) + a1_aug[0,l_] * W2[l,m] * H[0,m])
            Hass[l_m, nW_layer2+kl] = Hass[nW_layer2+kl, l_m]
    
    return Hass

def act(z, actFun):
    if actFun == 'tanh':
        return np.tanh(z)
    if actFun == 'rect':
        return np.maximum(z, 0)
        
def derivActFun(z, actFun, order): #input array, derivative of activation function 
    if actFun == 'tanh':#d tanh = 1-tanh^2, d^2 tanh = -2 * tanh * (1-tanh^2)
        a = np.tanh(z)
        if order == 1:
            res = 1-a*a
        if order == 2:
            res = -2*a*(1-a*a)
    if actFun == 'rect':
        if order == 1:
            res = np.ones(z.shape) * (z>0)
        if order == 2:
            res = np.zeros(z.shape)
    return res
    
def getPenalty(model, X, y):
    dataSize = len(X)
    deriv = getDerivative(model, X[0,:], y[0])
#    numParam = deriv.shape[0]
    J = deriv.dot(deriv.T)
    V = get2ndDerivative(model, X[0,:], y[0])
    for n in range(1, dataSize):
        deriv = getDerivative(model, X[n,:], y[n])
        J += deriv.dot(deriv.T)
        V += get2ndDerivative(model, X[n,:], y[n])
    J /= dataSize
    V /= (-dataSize)
    #print lin.eigvals(V), lin.eigvals(lin.inv(V)), lin.eigvals(J)
    return np.trace( lin.inv(V).dot(J) )
     
def main():
#%%    
    dataSize = 200
    X, y = generate_data(dataSize)
    actFun = 'tanh' #activation function: tanh or rect 
    num_passes=20000
    model = build_model(X, y, 3, num_passes, actFun, print_loss=True)
    visualize(X, y, model)
    pen = getPenalty(model, X, y)
    print pen/dataSize + calculate_loss(model, X, y)#5 seems the best
#%%

if __name__ == "__main__":
    main()
