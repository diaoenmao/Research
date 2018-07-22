import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from util import *
from modelWrapper import *
from modelselect import *

cudnn.benchmark = True
data_name = 'MNIST'
model_dir = 'mnist'
model_name = 'linear'
modelselect = 'penalty'
model_id = [10]
data_size = [5000]
modelselect_mode = ['AIC','BIC','GTIC']
metric = ['penalized_loss','loss','acc','modelselect_id']
TAG = data_name+'_'+model_name+'_'+modelselect
config.init()
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
save_mode = config.PARAM['save_mode']
init_seed = 0
num_Experiments = 1
seeds = list(range(init_seed,init_seed+num_Experiments))
input_feature = (28,28)
output_feature = 10
input_feature_idx = modelselect_input_feature(input_feature,init_size=2,step_size=1,start_point=None)

def main():
    result = []
    for i in range(len(seeds)):
        result.append(runExperiment('{}_{}'.format(seeds[i],TAG)))
    processResult(result,TAG)
    return
    
def runExperiment(Experiment_TAG):
    seed = int(Experiment_TAG.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    penalized_loss = {mode: torch.zeros((len(data_size),len(model_id)),device=device) for mode in modelselect_mode}
    loss = torch.zeros((len(data_size),len(model_id)),device=device)
    acc = torch.zeros((len(data_size),len(model_id)),device=device)
    modelselect_id = {mode: torch.zeros(len(data_size),device=device) for mode in modelselect_mode}
    for i in range(len(data_size)):
        train_dataset,test_dataset = fetch_dataset(data_name=data_name)
        train_loader,_,test_loader = split_dataset(train_dataset,test_dataset,data_size[i],0,0)
        
        for j in range(len(model_id)):
            cur_Experiment_TAG = Experiment_TAG+'_'+str(data_size[i])+'_'+str(model_id[j])
            cur_input_feature_idx = input_feature_idx[model_id[j]]

            model = eval('models.{}.{}(input_feature={},output_feature={}).to(device)'.format(model_dir,model_name,cur_input_feature_idx.shape[0],output_feature))
            criterion = nn.CrossEntropyLoss().to(device)
            mw = modelWrapper(model,config.PARAM['optimizer_name'])
            mw.set_optimizer_param(config.PARAM['optimizer_param'])
            mw.set_criterion(criterion)
            mw.set_optimizer()
                
            scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
            for epoch in range(max_num_epochs):
                scheduler.step()
                train_result = train(train_loader,mw,cur_input_feature_idx)
                test_result = test(test_loader,mw,cur_input_feature_idx)
                print_result(epoch,train_result,test_result)
                if(save_mode>1):
                    save([train_result,test_result],'./output/result/{}_{}'.format(cur_Experiment_TAG,epoch))

            loss[i,j] = test_result[2].avg
            acc[i,j] = test_result[3].avg


            irreduced_loss = torch.Tensor().to(device)
            mw.criterion.reduce = False
            for i, (input, target) in enumerate(train_loader):
                input, target = input.to(device), target.to(device)
                input = input.view(input.size(0),-1)[:,cur_input_feature_idx]
                output = mw.model(input)
                cur_loss = mw.loss(output,target)
                irreduced_loss = torch.cat([irreduced_loss,cur_loss])   
            for k in modelselect_mode:
                penalized_loss[k][i,j] = penalize(irreduced_loss,mw,k,cur_Experiment_TAG)
    for k in modelselect_mode:
        modelselect_id[k][:] = torch.argmin(penalized_loss[k],dim=1)
    result = {'penalized_loss':penalized_loss,'loss':loss,'acc':acc,'modelselect_id':modelselect_id}
    if(save_mode>0):    
        save(result,'./output/result/{}.pkl'.format(Experiment_TAG))        
    return result
    
def train(train_loader,mw,input_feature_idx):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    mw.model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input = input.view(input.size(0),-1)[:,input_feature_idx]
        data_time.update(time.time() - end)
        def closure():
            mw.optimizer.zero_grad()
            output = mw.model(input)
            loss = mw.loss(output,target)            
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = mw.acc(output, target, topk=(1, 5))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))
            loss.backward()
            return loss
        mw.optimizer.step(closure)
        batch_time.update(time.time() - end)
        end = time.time()
    return batch_time,data_time,losses,top1,top5
  
    
def test(validation_loader,mw,input_feature_idx):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    mw.model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(validation_loader):
            input, target = input.to(device), target.to(device)
            input = input.view(input.size(0),-1)[:,input_feature_idx]
            data_time.update(time.time() - end)
            output = mw.model(input)
            loss = mw.loss(output,target)
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = mw.acc(output, target, topk=(1, 5))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()              
    return batch_time,data_time,losses,top1,top5


def processResult(result,TAG):
    raw_result = np.zeros((num_Experiments,len(metric),len(modelselect_mode),len(data_size)))
    stat_result = {'Mean':{'penalized_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'acc':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_id':{mode: np.zeros(len(data_size)) for mode in modelselect_mode}},
                'Stderr':{'penalized_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'acc':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_id':{mode: np.zeros(len(data_size)) for mode in modelselect_mode}}}                  
    for i in range(num_Experiments):
        for j in range(len(metric)):
            for p in range(len(modelselect_mode)):
                if(metric[j]=='modelselect_id'):
                    raw_result[i,j,p,:] = result[i][metric[j]][modelselect_mode[p]].detach().int().numpy()
                else:               
                    for q in range(len(data_size)):
                        modelselect_id = result[i]['modelselect_id'][modelselect_mode[p]][q].detach().int().numpy()
                        if(metric[j]=='loss' or metric[j]=='acc'): 
                            raw_result[i,j,p,q] = result[i][metric[j]][q,modelselect_id].detach().numpy()
                        elif(metric[j]=='penalized_loss'):
                            raw_result[i,j,p,q] = result[i][metric[j]][modelselect_mode[p]][q,modelselect_id].detach().numpy()
    for j in range(len(metric)):               
        for p in range(len(modelselect_mode)):     
            stat_result['Mean'][metric[j]][modelselect_mode[p]] = np.mean(raw_result[:,j,p,:],axis=0)
            for q in range(len(data_size)):
                stat_result['Stderr'][metric[j]][modelselect_mode[p]][q] = np.std(raw_result[:,j,p,:],axis=0)/np.sqrt(data_size[q])
    all_result = {'raw_result':raw_result,'stat_result':stat_result}
    save(all_result,'./output/result/{}.pkl'.format(TAG))
    return all_result
    
if __name__ == "__main__":
    main()