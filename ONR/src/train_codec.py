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


cudnn.benchmark = True
data_name = 'ImageNet'
model_dir = 'imagenet'
model_name = 'cae'
milestones = [50,150,250]
data_size = 50000
metric = ['test_loss','timing']
TAG = data_name+'_'+model_name+'_'+modelselect
config.init()
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
save_mode = config.PARAM['save_mode']
init_seed = 0
num_Experiments = 1
seeds = list(range(init_seed,init_seed+num_Experiments))
input_feature = (128,128)

def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment('{}_{}'.format(seeds[i],TAG))
    return
    
    
def runExperiment(Experiment_TAG):
    seed = int(Experiment_TAG.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
       
    train_dataset,test_dataset = fetch_dataset(data_name=data_name)
    
    train_loader,validation_loader = split_dataset(train_dataset,test_dataset,data_size,0,1)
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    criterion_MSE = nn.MSELoss().to(device)
    mw = modelWrapper(model,config.PARAM['optimizer_name'])
    mw.set_optimizer_param(config.PARAM['optimizer_param'])
    mw.set_criterion(criterion_MSE)
    mw.set_optimizer()
    
    hyper_test_psnr = torch.zeros(max_num_epochs,device=device)
    scheduler = MultiStepLR(mw.optimizer, milestones=milestones, gamma=0.1)
    for epoch in range(max_num_epochs):
        scheduler.step()
        train_result = train(train_loader[0],mw)
        test_result = test(validation_loader[0],mw)
        print_result(epoch,train_result,test_result)
        hyper_test_psnr[epoch] = test_result[3].avg
    validated_num_epochs = torch.argmax(hyper_test_psnr) + 1
    print('Validated Number of Epoch: {}'.format(validated_num_epochs))
    
    
    train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,0,0)
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    criterion_MSE = nn.MSELoss().to(device)
    mw = modelWrapper(model,config.PARAM['optimizer_name'])
    mw.set_optimizer_param(config.PARAM['optimizer_param'])
    mw.set_criterion(criterion_MSE)
    mw.set_optimizer()
        
    scheduler = MultiStepLR(mw.optimizer, milestones=milestones, gamma=0.1)
    for epoch in range(validated_num_epochs):
        scheduler.step()
        train_result = train(train_loader,mw)
        test_result = test(test_loader,mw)
        print_result(epoch,train_result,test_result)
        if(save_mode>1):
            save([train_result,test_result],'./output/result/{}_{}'.format(cur_Experiment_TAG,epoch))
    timing = train_result[0].avg            
    test_loss = test_result[2].avg
    test_psnr = test_result[3].avg
    result = {'test_loss':test_loss,'test_psnr':test_psnr,'timing':timing}
    if(save_mode>0):
        save({'model_dict':mw.model.state_dict(),'optimizer_dict':optimizer.state_dict()},'./output/model/{}.pkl'.format(Experiment_TAG))  
        save(result,'./output/result/{}.pkl'.format(Experiment_TAG))        
    return result


def train(train_loader,mw):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    mw.model.train()
    end = time.time()
    for _, (input, _) in enumerate(train_loader):
        input = input.to(device)
        input = extract_patches_2D(input,input_feature)
        data_time.update(time.time() - end)
        def closure():
            mw.optimizer.zero_grad()
            output = mw.model(input)
            loss = mw.loss(input,output)
            psnr = PSNR(input,output[1])
            losses.update(loss.item(), input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            loss.backward()
            return loss
        mw.optimizer.step(closure)
        batch_time.update(time.time() - end)
        end = time.time()
    return batch_time,data_time,losses,psnrs
  
    
def test(validation_loader,mw):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    mw.model.eval()
    with torch.no_grad():
        end = time.time()
        for _, (input, _) in enumerate(validation_loader):
            input = input.to(device)
            input = extract_patches_2D(input,input_feature)
            data_time.update(time.time() - end)
            output = mw.model(input)
            loss = mw.loss(input,output)
            psnr = PSNR(input,output[1])
            losses.update(loss.item(), input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()              
    return batch_time,data_time,losses,psnrs
    