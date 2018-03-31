import theano
import theano.tensor as T
import numpy as np
from IO import *
import gtic
import gtic.mlp
import sys
import time
import os.path

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
     
def main():
    num_hidden_layers = 3
    data_init = 5000
    data_end = 10000
    max_nodes = 10
    dir = 'MLP.pkl'
    sys.setrecursionlimit(10000)
    ifload = False
    
    if(os.path.exists(dir) and ifload):
        tensors = load(dir)
        print("Load Complete") 
    else:
        start = time.time()
        tensors = gtic.mlp.BuildMLP(num_hidden_layers)
        end = time.time()
        print(("Last %fs" % (end - start)))
        print("Model Built")
        save(tensors, dir)
        print("Save Complete")

    start = time.time()
    input = [tensors[-2],tensors[-1]]
    tic =gtic.tic(*input)   
    result = theano.function(inputs=tensors[:-2],outputs=tic)
    end = time.time()
    print(("Last %fs" % (end - start)))
    print("TIC Built")

    while(True):
        print()
        #datasize = np.random.randint(data_init) + (data_end-data_init)
        datasize = 1000
        dim_in = np.random.randint(np.int(np.sqrt(datasize))) + 1
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
                param.append(np.random.randn(dim_hidden[i-1]+1,dim_out+1))
                num_totalparam = num_totalparam + (dim_hidden[i-1]+1)*1
            else:
                cur_hidden_nodes = np.random.randint(max_nodes) + 1 
                param.append(np.random.randn(dim_hidden[i-1]+1,cur_hidden_nodes))
                dim_hidden.append(cur_hidden_nodes)
                num_totalparam = num_totalparam + (dim_hidden[i-1]+1)*cur_hidden_nodes
        print(("DataSize: %d, theano.tensorotalParam: %d, Structure: dim_in: %d, dim_hidden: [%s], dim_out: %d" % (datasize, num_totalparam, dim_in, ', '.join(map(str, dim_hidden)),dim_out)))            
        
        start = time.time()
        input = [dataX, dataY]
        input.extend(param)  
        tic = result(*input)
        end = time.time()
        print(("Last %fs" % (end - start)))
        print("TIC: %f" % tic)
        print("TIC Evaluated")

if __name__ == "__main__":
    main()    