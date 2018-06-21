import shutil
import torch
import config
import time
import torch.backends.cudnn as cudnn
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from util import *
import models as models
from modelWrapper import *

data_name = 'MOSI'
model_dir = 'mosi'
model_name = 'lstm'
label_mode = 'binary'
TAG = data_name+'_'+model_name+'_'+label_mode
config.init()
batch_size = config.PARAM['batch_size']
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
verbose = config.PARAM['verbose']
if_resume = config.PARAM['if_resume']
if_show = config.PARAM['if_show']
num_Experiments = 1
input_feature = [46,74,300]
def from_label_mode():
    if(label_mode=='regression'):
        criterion = nn.L1Loss().to(device)
        output_feature = 1
    elif(label_mode=='binary'):
        criterion = nn.CrossEntropyLoss().to(device)
        #criterion = nn.MultiMarginLoss(p=1,margin=1).to(device)
        output_feature = 2        
    elif(label_mode=='seven'):
        criterion = nn.CrossEntropyLoss().to(device)
        #criterion = nn.MultiMarginLoss(p=1,margin=1).to(device)
        output_feature = 7
    return criterion,output_feature
    
criterion,output_feature = from_label_mode()
cudnn.benchmark = True
modality_name = ['visual','audio','text']

def main():
    if(if_resume):
        checkpoint = torch.load('./output/model/checkpoint.pth')
        init_seed = checkpoint['seed']
    else:
        init_seed = 0
    seeds = list(range(init_seed,init_seed+num_Experiments))
    for i in range(len(seeds)):
        runExperiment('{}_{}'.format(TAG,seeds[i]))
    return
    
def runExperiment(TAG):
    seed = TAG.split('_')[-1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    visual_train_loader,visual_valid_loader,visual_eval_loader,visual_test_loader = fetch_Multimodal_data(data_name,'visual',label_mode,batch_size)
    audio_train_loader,audio_valid_loader,audio_eval_loader,audio_test_loader = fetch_Multimodal_data(data_name,'audio',label_mode,batch_size)
    text_train_loader,text_valid_loader,text_eval_loader,text_test_loader = fetch_Multimodal_data(data_name,'text',label_mode,batch_size)
    combined_train_loader,combined_valid_loader,combined_eval_loader,combined_test_loader = fetch_Multimodal_data(data_name,'combined',label_mode,batch_size)   
    print('data ready')
    
    #train_loader,valid_loader,eval_loader,test_loader = visual_train_loader,visual_valid_loader,visual_eval_loader,visual_test_loader
    #train_loader,valid_loader,eval_loader,test_loader = audio_train_loader,audio_valid_loader,audio_eval_loader,audio_test_loader
    #train_loader,valid_loader,eval_loader,test_loader = text_train_loader,text_valid_loader,text_eval_loader,text_test_loader
    #train_loader,valid_loader,eval_loader,test_loader = combined_train_loader,combined_valid_loader,combined_eval_loader,combined_test_loader
    train_loader = [visual_train_loader,audio_train_loader,text_train_loader]
    valid_loader = [visual_valid_loader,audio_valid_loader,text_valid_loader]
    eval_loader = [visual_eval_loader,audio_eval_loader,text_eval_loader]
    test_loader = [visual_test_loader,audio_test_loader,text_test_loader]
    model_weight = np.zeros(3)
    print("Compute model weight")
    mw = create_mw()
    for i in range(len(modality_name)):
        print('Train '+ modality_name[i])
        model_weight[i] = train_modality(train_loader[i],valid_loader[i],mw[i],TAG+'_'+modality_name[i])
    if(output_feature>1):
        model_weight = np.exp(model_weight/np.max(model_weight))/np.sum(np.exp(model_weight/np.max(model_weight)))
    else:
        model_weight = np.exp(-model_weight)/np.sum(np.exp(-model_weight))
    print(model_weight)
    print("Retrain Model")
    mw = create_mw()
    for i in range(len(modality_name)):
        print('Train '+ modality_name[i])
        train_modality(eval_loader[i],test_loader[i],mw[i],TAG+'_'+modality_name[i])
    print("Evaluate Model")
    mw = create_mw()
    test_modality(test_loader,model_weight,mw,TAG)
    return

def create_mw():
    mw = []
    for i in range(len(modality_name)): 
        model = eval('models.{}.{}(input_feature={},output_feature={}).to(device)'.format(model_dir,model_name,input_feature[i],output_feature))
        mw.append(modelWrapper(model,config.PARAM['optimizer_name']))
        mw[i].set_optimizer_param(config.PARAM['optimizer_param'])
        mw[i].set_criterion(criterion)
        mw[i].wrap()
    return mw
    
def train_modality(train_loader,test_loader,mw,TAG):
    seed = TAG.split('_')[-2]
    if(if_resume):
        checkpoint = torch.load('./output/model/checkpoint_{}.pth'.format(TAG))
        init_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        best_epoch = checkpoint['best_epoch']
        mw.model.load_state_dict(checkpoint['state_dict'])
        mw.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint epoch {}"
            .format(checkpoint['epoch']))
    else:
        init_epoch = 0
        if(output_feature>1):
            best_prec1 = 0
        else:
            best_prec1 = 10
        best_epoch = 1 
        
    scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
    train_result = None
    test_result = None

    for epoch in range(init_epoch,init_epoch+max_num_epochs):
        scheduler.step()
        new_train_result = train(epoch,train_loader,mw)
        new_test_result = test(test_loader, mw)
        if(output_feature>1):
            prec1 = new_test_result[3].avg
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            prec1 = new_test_result[2].avg
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
        best_epoch = epoch if(is_best) else best_epoch
        print('Epoch: {0}\t'.format(epoch), end='')
        print_meter(new_test_result)           
        if(train_result is None):
            train_result = list(new_train_result)
            test_result = list(new_test_result)            
        else:
            for i in range(len(train_result)):
                train_result[i].merge(new_train_result[i])
                test_result[i].merge(new_test_result[i])
        save([train_result,test_result],'./output/result/{}_{}'.format(TAG,epoch))
        save_checkpoint({
            'seed': seed,
            'epoch': epoch,
            'state_dict': mw.model.state_dict(),
            'best_prec1': best_prec1,
            'best_epoch': best_epoch,
            'optimizer' : mw.optimizer.state_dict(),
        }, is_best,TAG)
        
    best = load('./output/model/best_{}.pth'.format(TAG))
    mw.model.load_state_dict(best['state_dict'])
    mw.optimizer.load_state_dict(best['optimizer'])    
    final_test_result = test(test_loader, mw)
    save(final_test_result,'./output/result/final_{}'.format(TAG))
    print('Test Result:')
    print_meter(final_test_result)
    if(output_feature>1):
        return final_test_result[3].avg
    else:
        return final_test_result[2].avg
   
def test_modality(test_loader,model_weight,mw,TAG):   
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    list_test_loader = []
    best = []
    for i in range(len(modality_name)):
        mw[i].model.eval()
        list_test_loader.append(list(test_loader[i]))
        best.append(torch.load('./output/model/best_{}.pth'.format(TAG+'_'+modality_name[i])))
    end = time.time()
    with torch.no_grad():
        num_batch = len(list_test_loader[0])
        for i in range(num_batch):
            output = 0
            for j in range(len(modality_name)):
                input, target = list_test_loader[j][i][0],list_test_loader[j][i][1]
                input, target = input.to(device), target.to(device)
                data_time.update(time.time() - end)              
                mw[j].model.load_state_dict(best[j]['state_dict'])
                if(model_name=='lstm'):
                    mw[j].model.hidden = mw[j].model.init_hidden(input.size(0))
                output += model_weight[j].item()*mw[j].model(input)
                batch_time.update(time.time() - end)
                end = time.time()                
            loss = mw[0].loss(output,target)           
            losses.update(loss.item(), input.size(0))
            if(output_feature>1):
                if(output_feature>5):
                    prec1, prec5 = mw[j].acc(output, target, topk=(1, 5))
                else:
                    prec1, prec5 = mw[j].acc(output, target, topk=(1, 1))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))
            else:
                top1.update(0, input.size(0))
                top5.update(0, input.size(0))
            if(verbose):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i+1, len(num_batch), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
    test_result = [batch_time,data_time,losses,top1,top5]     
    print('Test Result:')
    print_meter(test_result)
    return test_result[3].avg
    
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
        if(model_name=='lstm'):
            mw.model.hidden = mw.model.init_hidden(input.size(0))
        output = mw.model(input)
        loss = mw.loss(output,target)
        losses.update(loss.item(), input.size(0))
        if(output_feature>1):
            if(output_feature>5):
                prec1, prec5 = mw.acc(output, target, topk=(1, 5))
            else:
                prec1, prec5 = mw.acc(output, target, topk=(1, 1))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
        else:
            top1.update(0, input.size(0))
            top5.update(0, input.size(0))
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
            if(model_name=='lstm'):
                mw.model.hidden = mw.model.init_hidden(input.size(0))
            output = mw.model(input)
            loss = mw.loss(output,target)
            losses.update(loss.item(), input.size(0))
            if(output_feature>1):
                if(output_feature>5):
                    prec1, prec5 = mw.acc(output, target, topk=(1, 5))
                else:
                    prec1, prec5 = mw.acc(output, target, topk=(1, 1))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))
            else:
                top1.update(0, input.size(0))
                top5.update(0, input.size(0))
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


def save_checkpoint(state, is_best, TAG):
    filename='./output/model/checkpoint_{}.pth'.format(TAG)
    save(state, filename)
    if is_best:
        shutil.copyfile(filename, './output/model/best_{}.pth'.format(TAG))          
    return
    
def plt_result(TAG):
    best = load('./output/model/best_{}.pth'.format(TAG))
    best_prec1 = best['best_prec1']
    best_epoch = best['best_epoch']
    train_result,test_result = load('./output/result/{}_{}'.format(TAG,best_epoch))
    plt_meter([train_result,test_result],['train','test'],'{}_{}'.format(TAG,best_epoch))
    return
    
    
if __name__ == "__main__":
    main()