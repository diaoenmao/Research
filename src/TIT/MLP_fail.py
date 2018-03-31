import theano
import theano.tensor as T
import numpy as np
import time
import pickle
import sys


class HiddenLayer(object):
    def __init__(self, input, activation=T.tanh):
        self.input = input
        self.activation = activation
        
    def setParam(self, param_W, param_b):
        self.W = param_W
        self.b = param_b
    
    def setup(self,param_W,param_b):
        self.setParam(param_W,param_b)
        lin_output = T.dot(self.input, self.W) + self.b
        self.output=lin_output if self.activation is None else self.activation(lin_output)
        
class OutputLayer(object):
    def __init__(self, input, output, activation=T.tanh):
        self.input = input
        self.output = output
        self.activation = activation
        
    def setParam(self, param_W, param_b):
        self.W = param_W
        self.b = param_b
    
    def setup(self,param_W,param_b):
        self.setParam(param_W,param_b)
        if(True):
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
    def __init__(self, input, structure, output, activation=T.tanh):
        self.input = input
        self.structure = structure
        self.output = output
        self.activation = activation
        
    def setup(self, param):
        self.param = param
        self.hiddenlayers = []
        self.numtotalparam = 0
        result, updates = theano.scan(fn=self.reshapeParam,
                              outputs_info=None,
                              sequences=[T.arange(self.structure.shape[0]-1)])
        
        return result
        
        # for i in range(len(param)):   
            # cur_param = self.param[pivot:(pivot+T.prod(shapelist[i]))].reshape(shapelist[i])
            # cur_param_W = cur_param[1:,]
            # cur_param_b = cur_param[0,]  
            # if(i == 0):
                # self.hiddenlayers.append(HiddenLayer(self.input, self.activation))
                # self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
            # elif(i == len(param)-1):
                # self.outputlayer = OutputLayer(self.hiddenlayers[-1].output, self.output, self.activation)
                # self.outputlayer.setup(cur_param_W,cur_param_b)
            # else:
                # self.hiddenlayers.append(HiddenLayer(self.hiddenlayers[i-1].output, self.activation))
                # self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
            # pivot = pivot + T.prod(shapelist[i])
        # self.numtotalparam = pivot        
    
    def reshapeParam(self,i):
        print(i)
        theano.printing.debugprint(i)
        cur_dim_in = self.structure[i]
        cur_dim_out = self.structure[i+1]
        cur_shape = (cur_dim_in+1,cur_dim_out)
        cur_param = self.param[self.numtotalparam:(self.numtotalparam+T.prod(cur_shape))].reshape(cur_shape)
        cur_param_W = cur_param[1:,]
        cur_param_b = cur_param[0,]
        if(i == 0):
            print('a')
            self.hiddenlayers.append(HiddenLayer(self.input, self.activation))
            self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
        elif(i == self.structure.shape[0]-2):
            self.outputlayer = OutputLayer(self.hiddenlayers[-1].output, self.output, self.activation)
            self.outputlayer.setup(cur_param_W,cur_param_b)
        # else:
            # self.hiddenlayers.append(HiddenLayer(self.hiddenlayers[i-1].output, self.activation))
            # self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
        self.numtotalparam = self.numtotalparam + T.prod(cur_shape)
        return
        
    def negative_log_likelihood_vector(self):
        return(self.outputlayer.negative_log_likelihood_vector())
        
    def negative_log_likelihood_mean(self):
        return(self.outputlayer.negative_log_likelihood_mean())
                
    def TIC_gradient(self):        
        NLLV = self.negative_log_likelihood_vector()
        o=T.zeros((self.numtotalparam,self.numtotalparam))
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
        
        
def BuildMLP(num_hidden_layers):
    x = T.dmatrix('x')
    y = T.ivector('y')
    structure = T.ivector('struct')
    param = T.dvector('param')
    mlp = MLP(x, structure, y)
    mlp.setup(param)
    tic = mlp.TIC()
    returnedlist = [x,y,param,tic]
    return returnedlist

def BuildTIC(tensors):
    return theano.function(inputs=tensors[:-1],outputs=tensors[-1])

def TIC(TIC_model, dataX, dataY, param):
   input = [dataX, dataY]
   input.extend(param)
   return TIC_model(*input)

def saveModel(tensors, dir):
    pickle.dump(tensors, open(dir, "wb" ))
    return

def loadModel(dir):
    return pickle.load(open(dir, "rb" ))
     
def main():
    num_hidden_layers = 3
    data_init = 5000
    data_end = 10000
    max_nodes = 10
    dir = 'MLP.pkl'
    sys.setrecursionlimit(10000)

    start = time.time()
    tensors = BuildMLP(num_hidden_layers)
    end = time.time()
    print(end - start)
    print("Model Built")

    saveModel(tensors, dir)
    print("Save Complete")  

    start = time.time()
    TIC_model = BuildTIC(tensors)
    end = time.time()
    print(end - start)
    print("TIC Built")

    while(True):
        print()
        #datasize = np.random.randint(data_init) + (data_end-data_init)
        datasize = 1000
        dim_in = np.random.randint(np.int(np.sqrt(datasize)))
        dim_hidden = []
        dim_out = 1
        dataX = np.random.randn(datasize,dim_in)
        dataY = np.random.randint(2,size=datasize)
        num_totalparam = 0
        for i in range(num_hidden_layers+1):                   
            if(i==0):
                cur_hidden_nodes = np.random.randint(max_nodes) + 1 
                param = [np.random.randn(dim_in+1,cur_hidden_nodes)]
                dim_hidden.append(cur_hidden_nodes)
                num_totalparam = num_totalparam + (dim_in+1)*cur_hidden_nodes
            elif(i == num_hidden_layers):
                param.append(np.random.randn(dim_hidden[i-1]+1,1))
                num_totalparam = num_totalparam + (dim_hidden[i-1]+1)*1
            else:
                #cur_hidden_nodes = np.random.randint(dim_hidden[i-1]) + 1
                cur_hidden_nodes = np.random.randint(max_nodes) + 1 
                param.append(np.random.randn(dim_hidden[i-1]+1,cur_hidden_nodes))
                dim_hidden.append(cur_hidden_nodes)
                num_totalparam = num_totalparam + (dim_hidden[i-1]+1)*cur_hidden_nodes
        print(("DataSize: %d, TotalParam: %d, Structure: dim_in: %d, dim_hidden: [%s], dim_out: %d" % (datasize, num_totalparam, dim_in, ', '.join(map(str, dim_hidden)),dim_out)))            
        start = time.time()
        tic = TIC(TIC_model,dataX,dataY,param)
        end = time.time()
        print(end - start)
        print("TIC: %f" % tic)
        print("TIC Evaluated")

if __name__ == "__main__":
    main()   