import theano
import theano.tensor as T

def tic(loss, param):
    tic_hessian = theano.gradient.hessian(T.mean(loss),param)
    inverse = T.nlinalg.pinv(tic_hessian)
    o=T.zeros((param.shape[0],param.shape[0]))
    results, updates = theano.scan(lambda i, tmp: T.dot(T.grad(loss[i],param).reshape((-1,1)), T.grad(loss[i],param).reshape((-1,1)).T)+tmp,
              sequences=[T.arange(loss.shape[0])],
              outputs_info=[o])
    result = results[-1]
    tic_gradient = result/loss.shape[0] 
    tmp = T.nlinalg.trace(T.dot(inverse,tic_gradient))/loss.shape[0]
    return tmp  