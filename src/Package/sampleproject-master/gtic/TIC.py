import theano
import theano.tensor as T

class TIC(object):

    def __init__(self, loss, param):
        self.loss = loss
        self.param = param
        
    def tic_gradient(self):        
        o=T.zeros((self.param.shape[0],self.param.shape[0]))
        results, updates = theano.scan(lambda i, tmp: T.dot(T.grad(self.loss[i],self.param).reshape((-1,1)), T.grad(self.loss[i],self.param).reshape((-1,1)).T)+tmp,
                  sequences=[T.arange(self.loss.shape[0])],
                  outputs_info=[o])
        result = results[-1]
        out = result/self.loss.shape[0]
        return out
        
    def tic_hessian(self):
        H_theta = theano.gradient.hessian(T.mean(self.loss),self.param)
        return H_theta
        
    def tic(self):
        inverse = T.nlinalg.pinv(self.tic_hessian())
        tmp = T.nlinalg.trace(T.dot(inverse,self.tic_gradient()))/self.loss.shape[0]
        return tmp  