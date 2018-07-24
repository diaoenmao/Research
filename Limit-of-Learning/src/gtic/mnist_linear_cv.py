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
modelselect = 'cv'
model_id = [9,10]
data_size = [5000]
num_fold = [1]
metric = ['crossvalidation_loss','modelselect_loss','modelselect_acc','modelselect_id','efficiency','timing']
TAG = data_name+'_'+model_name+'_'+modelselect
config.init()
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
save_mode = config.PARAM['save_mode']
init_seed = 0
num_Experiments = 2
seeds = list(range(init_seed,init_seed+num_Experiments))
input_feature = (28,28)
output_feature = 10
input_feature_idx = modelselect_input_feature(input_feature,init_size=3,step_size=2,start_point=0)
#model_id = list(range(len(input_feature_idx))


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
    
    crossvalidation_loss = {str(n): torch.zeros((len(data_size),len(model_id)),device=device) for n in num_fold}
    test_loss = torch.zeros((len(data_size),len(model_id)),device=device)
    test_acc = torch.zeros((len(data_size),len(model_id)),device=device)
    modelselect_id = {str(n): torch.zeros(len(data_size),device=device) for n in num_fold}
    bestmodel_id = torch.zeros(len(data_size),device=device)
    timing = {str(n): torch.zeros((len(data_size),len(model_id)),device=device) for n in num_fold}
    for i in range(len(data_size)):
        train_dataset,test_dataset = fetch_dataset(data_name=data_name)
        for j in range(len(model_id)):
            cur_Experiment_TAG = Experiment_TAG+'_'+str(data_size[i])+'_'+str(model_id[j])
            cur_input_feature_idx = input_feature_idx[model_id[j]]

            model = eval('models.{}.{}(input_feature={},output_feature={}).to(device)'.format(model_dir,model_name,cur_input_feature_idx.shape[0],output_feature))
            criterion = nn.CrossEntropyLoss().to(device)
            mw = modelWrapper(model,config.PARAM['optimizer_name'])
            mw.set_optimizer_param(config.PARAM['optimizer_param'])
            mw.set_criterion(criterion)
            mw.set_optimizer()
            
            train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size[i],0,0)        
            scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
            for epoch in range(max_num_epochs):
                scheduler.step()
                train_result = train(train_loader,mw,cur_input_feature_idx)
                test_result = test(test_loader,mw,cur_input_feature_idx)
                print_result(epoch,train_result,test_result)
                if(save_mode>1):
                    save([train_result,test_result],'./output/result/{}_{}'.format(cur_Experiment_TAG,epoch))
            test_loss[i,j] = test_result[2].avg
            test_acc[i,j] = test_result[3].avg
            for n in num_fold:
                s = time.time()
                train_loader,validation_loader = split_dataset(train_dataset,test_dataset,data_size[i],0,n)
                cur_crossvalidation_loss = torch.zeros((n))
                for c in range(n):
                    model = eval('models.{}.{}(input_feature={},output_feature={}).to(device)'.format(model_dir,model_name,cur_input_feature_idx.shape[0],output_feature))
                    criterion = nn.CrossEntropyLoss().to(device)
                    mw = modelWrapper(model,config.PARAM['optimizer_name'])
                    mw.set_optimizer_param(config.PARAM['optimizer_param'])
                    mw.set_criterion(criterion)
                    mw.set_optimizer()
                        
                    scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
                    for epoch in range(max_num_epochs):
                        scheduler.step()
                        train_result = train(train_loader[c],mw,cur_input_feature_idx)
                        validation_result = test(validation_loader[c],mw,cur_input_feature_idx)                
                    cur_crossvalidation_loss[c] = validation_result[2].avg
                crossvalidation_loss[str(n)][i,j] = torch.mean(cur_crossvalidation_loss)
                e = time.time()
                timing[str(n)][i,j] = e-s
        for n in num_fold:
            modelselect_id[str(n)] = torch.argmin(crossvalidation_loss[str(n)],dim=1)
    bestmodel_id = torch.argmin(test_loss,dim=1)
    result = {'crossvalidation_loss':crossvalidation_loss,'test_loss':test_loss,'test_acc':test_acc,'modelselect_id':modelselect_id,'bestmodel_id':bestmodel_id,'timing':timing}
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


def processResult(result,TAG):
    raw_result = np.zeros((num_Experiments,len(metric),len(num_fold),len(data_size)),device=device)
    stat_result = {'Mean':{'crossvalidation_loss':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_loss':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_acc':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_id':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'efficiency':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'timing':{str(n): np.zeros(len(data_size)) for n in num_fold}},
                'Stderr':{'crossvalidation_loss':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_loss':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_acc':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'modelselect_id':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'efficiency':{str(n): np.zeros(len(data_size)) for n in num_fold},
                        'timing':{str(n): np.zeros(len(data_size)) for n in num_fold}}}
    for i in range(num_Experiments):
        for j in range(len(metric)):
            for p in range(len(num_fold)):
                if(metric[j]=='modelselect_id'):
                    raw_result[i,j,p,:] = result[i][metric[j]][str(num_fold[p])].detach().int().numpy()
                else:               
                    for q in range(len(data_size)):
                        modelselect_id = result[i]['modelselect_id'][str(num_fold[p])][q].detach().int().numpy()
                        if(metric[j]=='crossvalidation_loss'):
                            raw_result[i,j,p,q] = result[i]['crossvalidation_loss'][str(num_fold[p])][q,modelselect_id].detach().numpy()
                        if(metric[j]=='modelselect_loss'): 
                            raw_result[i,j,p,q] = result[i]['test_loss'][q,modelselect_id].detach().numpy()
                        elif(metric[j]=='modelselect_acc'): 
                            raw_result[i,j,p,q] = result[i]['test_acc'][q,modelselect_id].detach().numpy()
                        elif(metric[j]=='efficiency'):
                            bestmodel_id = result[i]['bestmodel_id'][q].detach().int().numpy()
                            raw_result[i,j,p,q] = result[i]['test_loss'][q,bestmodel_id].detach().numpy()/result[i]['test_loss'][q,modelselect_id].detach().numpy()
                        elif(metric[j]=='timing'):
                            raw_result[i,j,p,q] = torch.sum(result[i]['timing'][str(num_fold[p])][q,:]).detach().numpy()
    for j in range(len(metric)):               
        for p in range(len(num_fold)):     
            stat_result['Mean'][metric[j]][str(num_fold[p])] = np.mean(raw_result[:,j,p,:],axis=0)
            for q in range(len(data_size)):
                stat_result['Stderr'][metric[j]][str(num_fold[p])][q] = np.std(raw_result[:,j,p,q],axis=0)/np.sqrt(num_Experiments)
    print(result)
    print(raw_result)
    print(stat_result)
    all_result = {'raw_result':raw_result,'stat_result':stat_result}
    save(all_result,'./output/result/{}.pkl'.format(TAG))
    return all_result
    
if __name__ == "__main__":
    main()