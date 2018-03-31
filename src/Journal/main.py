import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from stochastic_runner import *
from full_runner import *


def main():   
   
    #mode = ['Base_1','AIC_1','BIC_1','BC_1','CrossValidation_1','CrossValidation_3','CrossValidation_10','GTIC_1','Lasso_1','Ridge_1','ElasticNet_1','GREG_1']
    mode = ['BIC_1']
    num_repetition = 1
    dataSize = [3000]

    runExperiment(dataSize,mode,num_repetition)


            
def runExperiment(dataSize,mode,num_repetition):
    seeds = list(range(num_repetition))
    seeds = [2]
    selected_model_id = np.zeros(len(seeds))
    best_model_id = np.zeros(len(seeds))
    selected_model_test_loss = np.zeros(len(seeds))
    best_model_test_loss = np.zeros(len(seeds))
    selected_model_test_acc = np.zeros(len(seeds))
    best_model_test_acc = np.zeros(len(seeds))
    efficiency = np.zeros(len(seeds))
    timing = np.zeros(len(seeds))
	for d in range(len(dataSize)):
		for m in range(len(mode)):
			for i in range(len(seeds)):
				s = time.time()
				randomGen = np.random.RandomState(seeds[i])
				X, y = fetch_data_logistic(dataSize,randomGen = randomGen)
				selected_model_id[i],best_model_id[i],selected_model_test_loss[i],best_model_test_loss[i],selected_model_test_acc[i],best_model_test_acc[i],efficiency[i] = Experiment(i,X,y,mode[m],randomGen)
				e = time.time()   
				timing[i] = e-s
				print('Total Elapsed Time for Experiment {}: {}'.format(i,timing[i])) 
				save([selected_model_id,best_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency,timing],'./output/Experiment_{}.pkl'.format(mode[m]))
			print('Total Elapsed Time for All {} Experiments : {}'.format(i+1,np.sum(timing))) 
    
def Experiment(id,X,y,mode,randomGen=None): 
    mode,K = mode.split('_')
    K = int(K)
    #optimizer_param = {'lr': 5*1e-2}
    optimizer_param = {'lr': 0.8}
    regularization_param = [0.001,0.001]
    optimizer_name = 'LBFGS'
    batch_size = 20
    ifcuda = True
    verbose = True
    ifsave = True
    ifshow = False
    ifregularize = False
    input_datatype = torch.FloatTensor
    target_datatype = torch.LongTensor
    max_num_epochs = 5
    min_delta = 5*1e-4
    patience = 5
    p = 0.1
    max_input_feature = np.int(np.sqrt(X.shape[0]*p))
    output_feature = 2
    print(max_input_feature)
    input_features = gen_input_features_LogisticRegression((max_input_feature,),start_point=[0])
    out_features = [output_feature]*len(input_features) 
    data = gen_data_LogisticRegression(X,y,K,p,input_features,randomGen)    
    models = gen_models_LogisticRegression(input_features,out_features,input_datatype,ifcuda)    
    modelwrappers = gen_modelwrappers(models,optimizer_param,optimizer_name)   
    criterion = nn.CrossEntropyLoss(reduce=False)   
    print('Start Experiment {} of {}_{}'.format(id,mode,K))      
    r = full_runner(id, data, modelwrappers, criterion, ifcuda, verbose, ifsave)
    #r = runner(id, data, modelwrappers, criterion, batch_size, ifcuda, verbose, ifsave)
    r.set_mode(mode)
    r.set_datatype(input_datatype,target_datatype)
    #r.set_early_stopping(max_num_epochs, min_delta, patience)
    r.set_max_num_epochs(max_num_epochs)
    r.set_regularization_param(ifregularize, regularization_param=regularization_param)
    r.train(ifshow)
    output = r.test()
    
    return output

    
if __name__ == "__main__":
    main()