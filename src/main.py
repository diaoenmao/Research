import torch
import copy
import config
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from stochastic_runner import *
from deterministic_runner import *
from modelWrapper import *

def main():   
   
    #mode = ['Base','AIC','BIC','BC','CrossValidation_1','CrossValidation_3','CrossValidation_10','CrossValidation_loo','GTIC','Lasso','Ridge','ElasticNet','GREG']
    config.init()
    mode = ['REG']
    num_Experiments = 1
    dataSize = [1000]

    run_Experiment(dataSize,mode,num_Experiments)
    mean,stderr = process_output(dataSize,mode)
    print(mean)
    print(stderr)


            
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
    remove_dir(['tmp','model','output'])
    for d in range(len(dataSize)):
        for m in range(len(mode)):
            for i in range(len(seeds)):
                s = time.time()
                randomGen = np.random.RandomState(seeds[i])
                #X, y = fetch_data_logistic(dataSize[d],randomGen = randomGen)
                #X, y = fetch_data_mld(dataSize[d],randomGen = randomGen) 
                X, y = fetch_data_circle(dataSize[d],randomGen = randomGen) 
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
    stderr = {k: np.zeros((len(dataSize),len(mode))) for k in keys}
    for d in range(len(dataSize)):
        for m in range(len(mode)):
            result = load('./result/Experiment_{}_{}.pkl'.format(dataSize[d],mode[m]))
            for i in range(len(keys)):
                mean[keys[i]][d,m] = np.mean(result[i])
                stderr[keys[i]][d,m] = np.std(result[i])/np.sqrt(result[i].shape[0])
    save([mean,stderr,dataSize,mode],'./result/final.pkl')
    return mean,stderr

def prepare_Linear(X,y,K,randomGen=None):
    init_size=None
    step_size=None
    start_point=[0]
    #start_point=None 
    dataSize = X.shape[0]   
    max_input_feature = np.int(np.sqrt(dataSize*config.PARAM['test_size'])) 
    dims = (max_input_feature,)
    #dims = (28,28)
    input_features = gen_input_features(dims,init_size=init_size,step_size=step_size,start_point=start_point)
    
    num_candidate_models = len(input_features)
    out_features = [config.PARAM['output_feature']]*num_candidate_models     
    models = gen_models_Linear(input_features,out_features,config.PARAM['input_datatype'],True,config.PARAM['ifcuda'])
    data = gen_data_Linear(X,y,K,config.PARAM['test_size'],input_features,randomGen)
    criterion = nn.CrossEntropyLoss(reduce=False)     
    modelwrappers = gen_modelwrappers(models,config.PARAM['optimizer_param'],config.PARAM['optimizer_name'],criterion,config.PARAM['regularization_parameters'],config.PARAM['if_joint_regularization'])     
    return data,modelwrappers

def prepare_MLP(X,y,K,randomGen=None):
    max_num_nodes = [20]
    init_size=[10]
    step_size=None
    hidden_layers = gen_hidden_layers(max_num_nodes,init_size=init_size,step_size=step_size)
    #hidden_layers = [(10,1,5),(10,20)]
    print(hidden_layers)
    
    num_candidate_models = len(hidden_layers)
    input_features = [X.shape[1]]*num_candidate_models
    out_features = [config.PARAM['output_feature']]*num_candidate_models   
    models = gen_models_MLP(input_features,hidden_layers,out_features,config.PARAM['input_datatype'],True,config.PARAM['ifcuda'])
    data = gen_data_Full(X,y,K,config.PARAM['test_size'],num_candidate_models,randomGen)
    criterion = nn.CrossEntropyLoss(reduce=False)     
    modelwrappers = gen_modelwrappers(models,config.PARAM['optimizer_param'],config.PARAM['optimizer_name'],criterion,config.PARAM['regularization_parameters'],config.PARAM['if_joint_regularization'])   
    return data,modelwrappers
    
def Experiment(id,X,y,mode,randomGen=None): 

    dataSize = X.shape[0]
    mode,K = parse_mode(mode,dataSize)
    
    #data,modelwrappers = prepare_Linear(X,y,K,randomGen)
    data,modelwrappers = prepare_MLP(X,y,K,randomGen)
    
    print('Start Experiment {} of {}_{}_{}'.format(id,dataSize,mode,K))      
    r = deterministic_runner(id, data, modelwrappers, config.PARAM['ifcuda'], config.PARAM['verbose'], config.PARAM['ifsave'])
    r.set_mode(mode)
    r.set_datatype(config.PARAM['input_datatype'],config.PARAM['target_datatype'])
    #r.set_early_stopping(max_num_epochs, min_delta, patience)
    r.set_max_num_epochs(config.PARAM['max_num_epochs'])
    r.set_ifregularize(config.PARAM['ifregularize'])
    r.train(config.PARAM['ifshow'])
    output = r.test()
    
    return output

    
if __name__ == "__main__":
    main()