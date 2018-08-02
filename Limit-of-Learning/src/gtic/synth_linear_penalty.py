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
data_name = 'SYNTH'
model_dir = 'synth'
model_name = 'linear'
modelselect = 'penalty'
model_id = [21,22]
data_size = [100,200]
milestones = [150]
modelselect_mode = ['AIC','BIC','GTIC']
metric = ['penalized_loss','modelselect_loss','modelselect_acc','modelselect_id','efficiency','timing']
TAG = data_name+'_'+model_name+'_'+modelselect
config.init()
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
save_mode = config.PARAM['save_mode']
init_seed = 0
num_Experiments = 2
seeds = list(range(init_seed,init_seed+num_Experiments))
input_feature = 50
output_feature = 2
input_feature_idx = modelselect_input_feature(input_feature,init_size=3,step_size=2,start_point=0)
#model_id = list(range(len(input_feature_idx))

def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment('{}_{}'.format(seeds[i],TAG))
    result = loadResult()
    processResult(result,TAG)
    return
    
def runExperiment(Experiment_TAG):
    seed = int(Experiment_TAG.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    penalized_loss = {mode: torch.zeros((len(data_size),len(model_id)),device=device) for mode in modelselect_mode}
    test_loss = torch.zeros((len(data_size),len(model_id)),device=device)
    test_acc = torch.zeros((len(data_size),len(model_id)),device=device)
    modelselect_id = {mode: torch.zeros(len(data_size),device=device) for mode in modelselect_mode}
    bestmodel_id = torch.zeros(len(data_size),device=device)
    timing = {mode: torch.zeros((len(data_size),len(model_id)),device=device) for mode in modelselect_mode}
    for i in range(len(data_size)):
        print('data size: {}'.format(data_size[i]))
        train_dataset,test_dataset = fetch_dataset_synth(input_feature*2,output_feature,randomGen=randomGen)
        train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size[i],0,0)
        
        for j in range(len(model_id)):
            print('model id: {}'.format(model_id[j]))
            cur_Experiment_TAG = Experiment_TAG+'_'+str(data_size[i])+'_'+str(model_id[j])
            cur_input_feature_idx = input_feature_idx[model_id[j]]

            model = eval('models.{}.{}(input_feature={},output_feature={}).to(device)'.format(model_dir,model_name,cur_input_feature_idx.shape[0],output_feature))
            criterion = nn.CrossEntropyLoss().to(device)
            mw = modelWrapper(model,config.PARAM['optimizer_name'])
            mw.set_optimizer_param(config.PARAM['optimizer_param'])
            mw.set_criterion(criterion)
            mw.set_optimizer()
                
            scheduler = MultiStepLR(mw.optimizer, milestones=milestones, gamma=0.1)
            for epoch in range(max_num_epochs):
                scheduler.step()
                train_result = train(train_loader,mw,cur_input_feature_idx)
                test_result = test(test_loader,mw,cur_input_feature_idx)
                print_result(epoch,train_result,test_result)
                if(save_mode>1):
                    save([train_result,test_result],'./output/result/{}_{}'.format(cur_Experiment_TAG,epoch))

            test_loss[i,j] = test_result[2].avg
            test_acc[i,j] = test_result[3].avg

            s1 = time.time()
            irreduced_loss = torch.Tensor().to(device)
            mw.criterion.reduction = 'none'
            for _, (input, target) in enumerate(train_loader):
                input, target = input.to(device), target.to(device)
                input = input.view(input.size(0),-1)[:,cur_input_feature_idx]
                output = mw.model(input)
                cur_loss = mw.loss(output,target)
                irreduced_loss = torch.cat([irreduced_loss,cur_loss])
            e1 = time.time()
            for k in modelselect_mode:
                s2 = time.time()
                penalized_loss[k][i,j] = penalize(irreduced_loss,mw,k,cur_Experiment_TAG)
                e2 = time.time()
                timing[k][i,j] = e1-s1+e2-s2
    for k in modelselect_mode:
        modelselect_id[k] = torch.argmin(penalized_loss[k],dim=1)
    bestmodel_id = torch.argmin(test_loss,dim=1)
    result = {'penalized_loss':penalized_loss,'test_loss':test_loss,'test_acc':test_acc,'modelselect_id':modelselect_id,'bestmodel_id':bestmodel_id,'timing':timing}
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
    for _, (input, target) in enumerate(train_loader):
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
        for _, (input, target) in enumerate(validation_loader):
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

def loadResult():
    result = []
    for i in range(num_Experiments):
        result.append(load('./output/result/{}_{}.pkl'.format(seeds[i],TAG)))
    return result
    
def processResult(result,TAG):
    print(result)
    raw_result = np.zeros((num_Experiments,len(metric),len(modelselect_mode),len(data_size)))
    stat_result = {'Mean':{'penalized_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_acc':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_id':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'efficiency':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'timing':{mode: np.zeros(len(data_size)) for mode in modelselect_mode}},
                'Stderr':{'penalized_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_loss':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_acc':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'modelselect_id':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'efficiency':{mode: np.zeros(len(data_size)) for mode in modelselect_mode},
                        'timing':{mode: np.zeros(len(data_size)) for mode in modelselect_mode}}}                        
    for i in range(num_Experiments):
        for j in range(len(metric)):
            for p in range(len(modelselect_mode)):
                if(metric[j]=='modelselect_id'):
                    raw_result[i,j,p,:] = result[i][metric[j]][modelselect_mode[p]].detach().int().cpu().numpy()
                else:               
                    for q in range(len(data_size)):
                        modelselect_id = result[i]['modelselect_id'][modelselect_mode[p]][q].detach().int().cpu().numpy()
                        if(metric[j]=='penalized_loss'):
                            raw_result[i,j,p,q] = result[i]['penalized_loss'][modelselect_mode[p]][q,modelselect_id].detach().cpu().numpy()
                        if(metric[j]=='modelselect_loss'): 
                            raw_result[i,j,p,q] = result[i]['test_loss'][q,modelselect_id].detach().cpu().numpy()
                        elif(metric[j]=='modelselect_acc'): 
                            raw_result[i,j,p,q] = result[i]['test_acc'][q,modelselect_id].detach().cpu().numpy()
                        elif(metric[j]=='efficiency'):
                            bestmodel_id = result[i]['bestmodel_id'][q].detach().int().cpu().numpy()
                            raw_result[i,j,p,q] = result[i]['test_loss'][q,bestmodel_id].detach().cpu().numpy()/result[i]['test_loss'][q,modelselect_id].detach().cpu().numpy()
                        elif(metric[j]=='timing'):
                            raw_result[i,j,p,q] = torch.sum(result[i]['timing'][modelselect_mode[p]][q,:]).detach().cpu().numpy()
    for j in range(len(metric)):               
        for p in range(len(modelselect_mode)):     
            stat_result['Mean'][metric[j]][modelselect_mode[p]] = np.mean(raw_result[:,j,p,:],axis=0)
            for q in range(len(data_size)):
                stat_result['Stderr'][metric[j]][modelselect_mode[p]][q] = np.std(raw_result[:,j,p,q],axis=0)/np.sqrt(num_Experiments)
    print(stat_result)
    all_result = {'raw_result':raw_result,'stat_result':stat_result}
    save(all_result,'./output/result/{}.pkl'.format(TAG))
    return all_result
    
if __name__ == "__main__":
    main()