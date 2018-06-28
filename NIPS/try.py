import theano
import theano.tensor as T
from theano import function


start = T.dmatrix('start')
t_nll = T.dvector('t_nll')
t_y = T.ivector('t_y')
t_nll = self.nll(t_y)
p=T.dvector('p')

results, _ = theano.scan(lambda result, t_nll, p: result + T.dot(T.grad(t_nll,p),T.grad(t_nll,p).T),
                         outputs_info=[start], sequences = [t_nll], non_sequences=[p])

f = function([t_y,p], results[-1])
f([y,self.param.get_value()])