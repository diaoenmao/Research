import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from stochastic_runner import *
from deterministic_runner import *


def main():   
   
    #mode = ['Base','AIC','BIC','BC','CrossValidation_1','CrossValidation_3','CrossValidation_10','CrossValidation_loo','GTIC','Lasso','Ridge','ElasticNet','GREG']
    mode = ['Base','AIC','BIC','BC','CrossValidation_1','CrossValidation_3','CrossValidation_10','CrossValidation_loo','GTIC']
    num_Experiments = 100
    dataSize = [1000,2000,3000,4000,5000]

    run_Experiment(dataSize,mode,num_Experiments)
    mean,std = process_output(dataSize,mode)
    print(mean)
    print(std)


            
def run_Experiment(dataSize,mode,num_Experiments):
    seeds = list(range(num_Experiments))
    selected_model_id = np.zeros(len(seeds))
    best_model_id = np.zeros(len(seeds))
    selected_model_test_loss = np.zeros(len(seeds))
    best_model_test_loss = np.zeros(len(seeds))
    selected_model_test_acc = np.zeros(len(seeds))
    best_model_test_acc = np.zeros(len(seeds))
    efficiency = np.zeros(len(seeds))
    timing = np.zeros(len(seeds))
    remove_dir(['data/stats','model','output'])
    for d in range(len(dataSize)):
        for m in range(len(mode)):
            for i in range(len(seeds)):
                s = time.time()
                randomGen = np.random.RandomState(seeds[i])
                X, y = fetch_data_logistic(dataSize[d],randomGen = randomGen)
                selected_model_id[i],best_model_id[i],selected_model_test_loss[i],best_model_test_loss[i],selected_model_test_acc[i],best_model_test_acc[i],efficiency[i] = Experiment(i,X,y,mode[m],randomGen)
                e = time.time()   
                timing[i] = e-s
                print('Total Elapsed Time for Experiment {}: {}'.format(i,timing[i])) 
                save([selected_model_id,best_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency,timing],'./output/Experiment_{}_{}.pkl'.format(dataSize[d],mode[m]))
            print('Total Elapsed Time for All {} Experiments : {}'.format(i+1,np.sum(timing)))
            save([selected_model_id,best_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency,timing],'./result/Experiment_{}_{}.pkl'.format(dataSize[d],mode[m]))
        zip_dir(['model','output'],'tmp/Experiment_{}.zip'.format(dataSize[d]))
        remove_dir(['model','output'])
        
def parse_mode(mode,dataSize):
    if('_' in mode):
        mode,K = mode.split('_')
        if(K!='loo'): 
            K = np.int(K)
    else:
        mode = mode
        K = 1
    return mode,K

def process_output(dataSize,mode):
    keys = ['selected_model_id','best_model_id','selected_model_test_loss','best_model_test_loss','selected_model_test_acc','best_model_test_acc','efficiency','timing']
    mean = {k: np.zeros((len(dataSize),len(mode))) for k in keys}
    std = {k: np.zeros((len(dataSize),len(mode))) for k in keys}
    for d in range(len(dataSize)):
        for m in range(len(mode)):
            result = load('./result/Experiment_{}_{}.pkl'.format(dataSize[d],mode[m]))
            for i in range(len(keys)):
                mean[keys[i]][d,m] = np.mean(result[i])
                std[keys[i]][d,m] = np.std(result[i])
    save([mean,std,dataSize,mode],'./result/final.pkl')
    return mean,std
    
def Experiment(id,X,y,mode,randomGen=None): 
    dataSize = X.shape[0]
    mode,K = parse_mode(mode,dataSize)
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
    max_input_feature = np.int(np.sqrt(dataSize*p))
    output_feature = 2
    print(max_input_feature)
    input_features = gen_input_features_Linear((max_input_feature,),start_point=[0])
    out_features = [output_feature]*len(input_features) 
    data = gen_data_Linear(X,y,K,p,input_features,randomGen)    
    models = gen_models_Linear(input_features,out_features,input_datatype,True,ifcuda)    
    modelwrappers = gen_modelwrappers(models,optimizer_param,optimizer_name)   
    criterion = nn.CrossEntropyLoss(reduce=False)   
    print('Start Experiment {} of {}_{}_{}'.format(id,dataSize,mode,K))      
    r = deterministic_runner(id, data, modelwrappers, criterion, ifcuda, verbose, ifsave)
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