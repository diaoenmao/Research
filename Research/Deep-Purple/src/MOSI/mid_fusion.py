import shutil
import torch
import config
import time
import copy
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
model_name = 'conv'
label_mode = 'seven'
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
hidden_feature = {'conv':64, 'lstm':128}
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
merge_mode = 'concat'

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
    if(merge_mode=='average'):
        model_weight = np.zeros(3)
        print("Compute model weight")
        mw = create_mw()
        for i in range(len(modality_name)):
            print('Train '+ modality_name[i])
            model_weight[i] = train_modality(train_loader[i],valid_loader[i],mw[i],TAG+'_'+modality_name[i])
        model_weight = model_weight/np.sum(model_weight)
        print(model_weight)
    
    print("Train Model")
    mw = create_mw()
    for i in range(len(modality_name)):
        print('Train '+ modality_name[i])
        train_modality(eval_loader[i],test_loader[i],mw[i],TAG+'_'+modality_name[i])
        
    mw = merge_mw(merge_mode,TAG)
    print("Test Merged Model")
    merged_test_result = test(combined_test_loader, mw)
    print('Test Result:')
    print_meter(merged_test_result)
    save(merged_test_result,'./output/result/merged_{}'.format(TAG))
    
    print("Retrain Full Model")       
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
        best_prec1 = 0
        best_epoch = 1 
        
    scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
    train_result = None
    test_result = None

    for epoch in range(init_epoch,init_epoch+max_num_epochs):
        scheduler.step()
        new_train_result = train(epoch,combined_eval_loader,mw)
        new_test_result = test(combined_test_loader, mw)
        prec1 = new_test_result[3].avg
        is_best = prec1 > best_prec1
        best_epoch = epoch if(is_best) else best_epoch
        best_prec1 = max(prec1, best_prec1)
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
    final_test_result = test(combined_test_loader, mw)
    save(final_test_result,'./output/result/final_{}'.format(TAG))
    print('Test Result:')
    print_meter(final_test_result)
    
    
    exit()
    print("Evaluate Model")
    mw = create_mw()
    return

def create_mw():
    mw = []
    for i in range(len(modality_name)): 
        model = eval('models.{}.{}(input_feature={},output_feature={},hidden_feature={}).to(device)'.format(model_dir,model_name,input_feature[i],output_feature,hidden_feature[model_name]))
        mw.append(modelWrapper(model,config.PARAM['optimizer_name']))
        mw[i].set_optimizer_param(config.PARAM['optimizer_param'])
        mw[i].set_criterion(criterion)
        mw[i].wrap()
    return mw


def merge_mw(mode,TAG,weight=None):
    model = eval('models.{}.{}(input_feature={},output_feature={},hidden_feature={}).to(device)'.format(model_dir,model_name,sum(input_feature),output_feature,len(modality_name)*hidden_feature[model_name]))
    target_mw = modelWrapper(model,config.PARAM['optimizer_name'])
    target_dict = target_mw.model.state_dict()
    if(mode=='average'):
        copy_target_dict = copy.deepcopy(target_dict)
        for k,_ in copy_target_dict.items():
            copy_target_dict[k].fill_(0)
        if(weight is None):
            weight = np.ones(len(modality_name))/len(modality_name)       
        for i in range(len(modality_name)):
            best = torch.load('./output/model/best_{}.pth'.format(TAG+'_'+modality_name[i]))            
            for k,v in best['state_dict'].items():
                if(copy_target_dict[k].dim()>1 and k.split('.')[0]!='classifier'):
                    if(copy_target_dict[k].size()==best['state_dict'][k].size()):
                        copy_target_dict[k] += torch.tensor(weight[i],device=device)*best['state_dict'][k]
                    else:
                        copy_target_dict[k] = target_dict[k]
        target_mw.model.load_state_dict(copy_target_dict)
    elif(mode=='concat'):          
        for k,v in target_dict.items():
            tracker = [0,0]
            if(target_dict[k].dim()>1 and k.split('.')[0]!='classifier'):
                for i in range(len(modality_name)):
                    best = torch.load('./output/model/best_{}.pth'.format(TAG+'_'+modality_name[i]))  
                    output_span,input_span =  best['state_dict'][k].size(0),best['state_dict'][k].size(1)
                    target_dict[k][tracker[0]:tracker[0]+output_span,tracker[1]:tracker[1]+input_span,] = best['state_dict'][k]
                    tracker[0],tracker[1] = tracker[0]+output_span,tracker[1]+input_span                  
        target_mw.model.load_state_dict(target_dict)
    else:
        error('merge mode not supported')
    target_mw.set_optimizer_param(config.PARAM['optimizer_param'])
    target_mw.set_criterion(criterion)
    target_mw.wrap()
    return target_mw
    
    
    
    
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
        best_prec1 = 0
        best_epoch = 1 
        
    scheduler = MultiStepLR(mw.optimizer, milestones=[150], gamma=0.1)
    train_result = None
    test_result = None

    for epoch in range(init_epoch,init_epoch+max_num_epochs):
        scheduler.step()
        new_train_result = train(epoch,train_loader,mw)
        new_test_result = test(test_loader, mw)
        prec1 = new_test_result[3].avg
        is_best = prec1 > best_prec1
        best_epoch = epoch if(is_best) else best_epoch
        best_prec1 = max(prec1, best_prec1)
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
    return final_test_result[3].avg
   
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
            if(output_feature>5):
                prec1, prec5 = mw[0].acc(output, target, topk=(1, 5))
            else:
                prec1, prec5 = mw[0].acc(output, target, topk=(1, 1))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
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
        if(output_feature>5):
            prec1, prec5 = mw.acc(output, target, topk=(1, 5))
        else:
            prec1, prec5 = mw.acc(output, target, topk=(1, 1))
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
            if(model_name=='lstm'):
                mw.model.hidden = mw.model.init_hidden(input.size(0))
            output = mw.model(input)
            loss = mw.loss(output,target)
            losses.update(loss.item(), input.size(0))
            if(output_feature>5):
                prec1, prec5 = mw.acc(output, target, topk=(1, 5))
            else:
                prec1, prec5 = mw.acc(output, target, topk=(1, 1))
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