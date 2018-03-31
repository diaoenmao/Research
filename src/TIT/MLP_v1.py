import theano
import theano.tensor as T
import numpy as np
import time

class HiddenLayer(object):
    def __init__(self, input, dim_in, dim_out, activation=T.tanh):
        self.input = input
        self.activation = activation
        self.dim_in = dim_in
        self.dim_out = dim_out
        
    def setParam(self, param_W, param_b):
        self.W = param_W
        self.b = param_b
    
    def setup(self,param_W,param_b):
        self.setParam(param_W,param_b)
        lin_output = T.dot(self.input, self.W) + self.b
        self.output=lin_output if self.activation is None else self.activation(lin_output)
        
class OutputLayer(object):
    def __init__(self, input, output, dim_in, dim_out, activation=T.tanh):
        self.input = input
        self.output = output
        self.activation = activation
        self.dim_in = dim_in
        self.dim_out = dim_out
        
    def setParam(self, param_W, param_b):
        self.W = param_W
        self.b = param_b
    
    def setup(self,param_W,param_b):
        self.setParam(param_W,param_b)
        if(self.dim_out==1):
            self.p_y_given_x = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
            self.p_y_given_x = T.concatenate([(1-self.p_y_given_x),(self.p_y_given_x)],axis=1)
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        
    def negative_log_likelihood_vector(self):
        tmp = self.p_y_given_x[T.arange(self.input.shape[0]), self.output]
        return -T.log(tmp)
                
    def negative_log_likelihood_mean(self):
        return T.mean(self.negative_log_likelihood_vector())
        
        
class MLP(object):
    def __init__(self, input, output, dim_in, dim_hidden, dim_out, activation=T.tanh):
        self.input = input
        self.output = output
        self.activation = activation
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
    def setup(self, param):
        self.param = param
        self.hiddenlayers = []
        pivot = 0
        self.numtotalparam = 0
        for i in range(len(self.dim_hidden)):
            if(i == 0):
                cur_dim_in = self.dim_in + 1
                cur_dim_out = self.dim_hidden[i]
                self.hiddenlayers.append(HiddenLayer(self.input, cur_dim_in, cur_dim_out, self.activation))
            else:
                cur_dim_in = self.dim_hidden[i-1] + 1
                cur_dim_out = self.dim_hidden[i]
                self.hiddenlayers.append(HiddenLayer(self.hiddenlayers[i-1].output, cur_dim_in, cur_dim_out, activation=T.tanh))
            cur_numparam = cur_dim_in*cur_dim_out
            cur_param = self.param[pivot:(pivot+cur_numparam)].reshape((cur_dim_in, cur_dim_out))
            cur_param_W = cur_param[1:cur_dim_in,]
            cur_param_b = cur_param[0,]
            self.hiddenlayers[i].setup(cur_param_W,cur_param_b)                    
            self.numtotalparam = self.numtotalparam + cur_numparam
            pivot = pivot + cur_numparam
        cur_dim_in = self.dim_hidden[-1] + 1
        cur_dim_out = self.dim_out
        self.outputlayer = OutputLayer(self.hiddenlayers[-1].output, self.output, cur_dim_in, cur_dim_out, self.activation)
        cur_numparam = cur_dim_in*cur_dim_out
        cur_param = self.param[pivot:(pivot+cur_numparam)].reshape((cur_dim_in, cur_dim_out))
        cur_param_W = cur_param[1:cur_dim_in,]
        cur_param_b = cur_param[0,]
        self.outputlayer.setup(cur_param_W,cur_param_b)                    
        self.numtotalparam = self.numtotalparam + cur_numparam
            
    def negative_log_likelihood_vector(self):
        return(self.outputlayer.negative_log_likelihood_vector())
        
    def negative_log_likelihood_mean(self):
        return(self.outputlayer.negative_log_likelihood_mean())
                
    def TIC_gradient(self):        
        NLLV = self.negative_log_likelihood_vector()
        o=np.zeros((self.numtotalparam,self.numtotalparam))
        results, updates = theano.scan(lambda i, tmp: T.dot(T.grad(NLLV[i],self.param).reshape((-1,1)), T.grad(NLLV[i],self.param).reshape((-1,1)).T)+tmp,
                  sequences=[T.arange(self.input.shape[0])],
                  outputs_info=[o])
        result = results[-1]
        out = result/self.input.shape[0]
        return out
        
    def TIC_hessian(self):
        NLLM = self.negative_log_likelihood_mean()
        H_theta = theano.gradient.hessian(NLLM,self.param)
        return H_theta
        
    def TIC(self):
        inverse = T.nlinalg.pinv(self.TIC_hessian())
        tmp = T.nlinalg.trace(T.dot(inverse,self.TIC_gradient()))/self.input.shape[0]
        return tmp        
        
        
def BuildMLP(dim_in, dim_hidden, dim_out):
    x = T.dmatrix('x')
    y = T.ivector('y')
    param = T.dvector('param')
    mlp = MLP(x, y, dim_in, dim_hidden, dim_out)
    mlp.setup(param)
    tic = mlp.TIC()
    # start = time.time()
    # TIC_model = theano.function(inputs=[x,y,param],outputs=tic)
    # end = time.time()
    # print(end - start)
    #return [x,y,param,tic,mlp]
    return [x,y,param,tic]

def main():
   dim_in, dim_hidden, dim_out = 2,[3],1
   result = BuildMLP(dim_in, dim_hidden, dim_out)
   print("After Compile")
   start = time.time()
   TIC_model = theano.function(inputs=[result[0],result[1],result[2]],outputs=result[3])
   end = time.time()
   print(end - start)
   dataX = np.random.randn(10,2)
   dataY = np.random.randint(2,size=10)
   param = np.random.randn(13)
   print(TIC_model(dataX,dataY,param))

if __name__ == "__main__":
    main()   