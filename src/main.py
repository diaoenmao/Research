import shutil
import torch
import config
import time
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from data import *
from util import *
from model import *
from modelWrapper import *

data_name = 'CIFAR10'
model_name = 'AlexNet'
TAG = data_name+'_'+model_name
config.init()
batch_size = config.PARAM['batch_size']
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
verbose = config.PARAM['verbose']
if_resume = config.PARAM['if_resume']
num_Experiments = 1
output_feature = 10

def main():
    if(if_resume):
        checkpoint = torch.load('./model/checkpoint.pth')
        init_seed = checkpoint['seed']
    else:
        remove_dir(['model','output'])
        init_seed = 0
    seeds = list(range(init_seed,init_seed+num_Experiments))
    for i in range(len(seeds)):
        runExperiment(seeds[i],'{}_{}'.format(TAG,i))
    return
    
def runExperiment(seed,TAG):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_loader,test_loader = fetch_data(data_name=data_name,batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss().to(device)
    model = eval('{}(num_classes={}).to(device)'.format(model_name,output_feature))
    mw = modelWrapper(model,config.PARAM['optimizer_name'])
    mw.set_optimizer_param(config.PARAM['optimizer_param'])
    mw.set_criterion(criterion)
    mw.wrap()

    if(if_resume):
        checkpoint = torch.load('./model/checkpoint.pth')
        init_epoch = checkpoint['epoch'] + 1
        best_prec1 = checkpoint['best_prec1']
        mw.model.load_state_dict(checkpoint['state_dict'])
        mw.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint epoch {}"
            .format(checkpoint['epoch']))
    else:
        init_epoch = 0
        best_prec1 = 0
    
    scheduler = StepLR(mw.optimizer, step_size=30, gamma=0.1)
    train_result = None
    test_result = None  
    for epoch in range(init_epoch,init_epoch+max_num_epochs):
        scheduler.step()
        new_train_result = train(epoch,train_loader,mw)
        new_test_result = test(test_loader, mw)
        prec1 = new_test_result[3].avg
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Epoch: {0}\t'
              'Loss {losses.avg:.4f}\t'
              'Prec@1 {prec1.avg:.3f}\t'
              'Prec@5 {prec5.avg:.3f}\t'
              'Time {time}\t'
              .format(epoch,losses=new_test_result[2],prec1=new_test_result[3],prec5=new_test_result[4],time=new_train_result[0].avg*len(train_loader)))              
        if(train_result is None):
            train_result = list(new_train_result)
            test_result = list(new_test_result)            
        else:
            for i in range(len(train_result)):
                train_result[i].merge(new_train_result[i])
                test_result[i].merge(new_test_result[i])
        save([train_result,test_result],'./output/{}_{}'.format(TAG,epoch+1))
        save_checkpoint({
            'seed': seed,
            'epoch': epoch,
            'state_dict': mw.model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : mw.optimizer.state_dict(),
        }, is_best)
    
def train(epoch,train_loader, mw):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    mw.model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        data_time.update(time.time() - end)
        output = mw.model(input)
        loss = mw.loss(output,target)
        losses.update(loss.item(), input.size(0))
        prec1, prec5 = mw.acc(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        mw.optimizer.zero_grad()
        loss.backward()
        mw.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if(verbose):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
                  
    return batch_time,data_time,losses,top1,top5
  
    
def test(val_loader, mw):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    mw.model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            data_time.update(time.time() - end)
            output = mw.model(input)
            loss = mw.loss(output,target)
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = mw.acc(output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if(verbose):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i+1, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
              
    return batch_time,data_time,losses,top1,top5


def save_checkpoint(state, is_best, filename='./model/checkpoint.pth'):
    save(state, filename)
    if is_best:
        shutil.copyfile(filename, './model/best.pth')          
    return
    
    
if __name__ == "__main__":
    main()