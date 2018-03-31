import theano
import theano.tensor as T

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
        self.p_y_given_x = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        self.p_y_given_x = T.concatenate([(1-self.p_y_given_x),(self.p_y_given_x)],axis=1)
        #self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        
    def negative_log_likelihood_vector(self):
        tmp = self.p_y_given_x[T.arange(self.input.shape[0]), self.output]
        return -T.log(tmp)
                
    def negative_log_likelihood_mean(self):
        return T.mean(self.negative_log_likelihood_vector())
        
        
class MLP(object):
    def __init__(self, input, output, activation=T.tanh):
        self.input = input
        self.output = output
        self.activation = activation
        
    def setup(self, param):
        paramlist = []
        shapelist = []
        self.hiddenlayers = []
        for i in range(len(param)):
            cur_flatten_param = T.flatten(param[i],1)
            if(i==len(param)-1):
                self.last_layer_param = cur_flatten_param
            paramlist.append(cur_flatten_param)
            shapelist.append(param[i].shape)
        self.param = T.concatenate(paramlist,axis=0)
        self.numtotalparam = 0
        for i in range(len(param)):   
            cur_param = self.param[self.numtotalparam:(self.numtotalparam+T.prod(shapelist[i]))].reshape(shapelist[i])
            cur_param_W = cur_param[1:,]
            cur_param_b = cur_param[0,]  
            if(i == 0):
                self.hiddenlayers.append(HiddenLayer(self.input, self.activation))
                self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
            elif(i == len(param)-1):
                self.outputlayer = OutputLayer(self.hiddenlayers[-1].output, self.output, self.activation)
                self.outputlayer.setup(cur_param_W,cur_param_b)
            else:
                self.hiddenlayers.append(HiddenLayer(self.hiddenlayers[i-1].output, self.activation))
                self.hiddenlayers[i].setup(cur_param_W,cur_param_b)
            self.numtotalparam = self.numtotalparam + T.prod(shapelist[i])       
    
    def getParam(self):
        return self.param
        
    def getLastLayerParam(self):
        return self.last_layer_param
        
    def negative_log_likelihood_vector(self):
        return(self.outputlayer.negative_log_likelihood_vector())
        
    def negative_log_likelihood_mean(self):
        return(self.outputlayer.negative_log_likelihood_mean())      
        
        
def BuildMLP(num_hidden_layer,last_layer = False):
    x = T.dmatrix('x')
    y = T.lvector('y')
    param = []
    for i in range(num_hidden_layer+1):
        param.append(T.dmatrix(('param%d' % i)))
    mlp = MLP(x, y)
    mlp.setup(param)
    loss = mlp.negative_log_likelihood_vector()
    tensors = [x,y]
    tensors.extend(param)
    if(last_layer):        
        last_layer_flattenparam = mlp.getLastLayerParam()
        tensors.extend([loss,last_layer_flattenparam])  
    else:
        flattenparam = mlp.getParam()
        tensors.extend([loss,flattenparam])   
    return tensors

def BuildModelClass(num_hidden_layers,last_layer = False):
    tensorslist = []
    for i in range(len(num_hidden_layers)):
        num_hidden_layer = num_hidden_layers[i]
        tensorslist.append(BuildMLP(num_hidden_layer,last_layer))
    return tensorslist