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
data_name = 'MNIST'
model_dir = 'mnist'
model_name = 'CAE'
TAG = data_name+'_'+model_name
config.init()
milestones = config.PARAM['milestones']
gamma = config.PARAM['gamma']
patch_shape = config.PARAM['patch_shape']
batch_size = config.PARAM['batch_size']
data_size = config.PARAM['data_size']
device = torch.device(config.PARAM['device'])    
max_num_epochs = config.PARAM['max_num_epochs']
save_mode = config.PARAM['save_mode']
init_seed = 0
num_Experiments = 1
seeds = list(range(init_seed,init_seed+num_Experiments))

def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment('{}_{}'.format(seeds[i],TAG))
    return
    
    
def runExperiment(Experiment_TAG):
    seed = int(Experiment_TAG.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    train_dataset,test_dataset = fetch_dataset(data_name=data_name)
    validated_num_epochs = max_num_epochs
    
    train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,batch_size=batch_size,num_fold=0,radomGen=randomGen)
    print('Training data size {}, Test data size {}'.format(len(train_dataset),len(test_dataset)))
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    summary(model.to('cuda'), input_size=(1, patch_shape[0], patch_shape[1]))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    mw = modelWrapper(model,config.PARAM['optimizer_name'])
    mw.set_optimizer_param(config.PARAM['optimizer_param'])
    mw.set_criterion(criterion)
    mw.set_optimizer()
        
    scheduler = MultiStepLR(mw.optimizer, milestones=milestones, gamma=gamma)
    for epoch in range(validated_num_epochs):
        scheduler.step()
        train_result = train(train_loader,mw,epoch)
        test_result = test(test_loader,mw,epoch)
        print_result(epoch,train_result,test_result)
        if(save_mode>1):
            save([train_result,test_result],'./output/result/{}_{}'.format(cur_Experiment_TAG,epoch))
    timing = train_result[0].avg            
    test_loss = test_result[2].avg
    test_psnr = test_result[3].avg
    test_acc = test_result[4].avg
    result = {'test_loss':test_loss,'test_psnr':test_psnr,'test_acc':test_acc,'timing':timing}
    if(save_mode>0):
        save({'model_dict':mw.model.state_dict(),'optimizer_dict':mw.optimizer.state_dict()},'./output/model/{}.pkl'.format(Experiment_TAG))  
        save(result,'./output/result/{}.pkl'.format(Experiment_TAG))        
    return result
    
def train(train_loader,mw,epoch):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    accs = Meter()
    mw.model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        data_time.update(time.time() - end)
        output = mw.model(input)
        loss = mw.loss(output,target)
        psnr = PSNR(output[0],input)
        acc = ACC(output[1],target,topk=(1,))
        losses.update(loss.item(), input.size(0))
        psnrs.update(psnr.item(), input.size(0))
        accs.update(acc[0], input.size(0))
        mw.optimizer.zero_grad()
        loss.backward()
        mw.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % (len(train_loader)//5) == 0:
            print('Train Epoch: {}[({:.0f}%)]\tLoss: {:.4f}\tPNSR: {:.4f}\tACC: {:.4f}'.format(
                epoch, 100. * i / len(train_loader), loss.item(), psnr.item(), acc[0]))
    return batch_time,data_time,losses,psnrs,accs
  
    
def test(validation_loader,mw,epoch):
    output_dir = './output/img'
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    accs = Meter()
    mw.model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(validation_loader):
            input = input.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            output = mw.model(input)
            loss = mw.loss(output,target)
            psnr = PSNR(output[0],input)
            acc = ACC(output[1],target,topk=(1,))
            losses.update(loss.item(), input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            accs.update(acc[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        if epoch % 3 == 0:
            if not os.path.exists('.{}/image_{}.png'.format(output_dir,i)):
                save_img(input,'{}/image_{}.png'.format(output_dir,i))
            save_img(output[0],'{}/image_{}_{}.png'.format(output_dir,i,epoch))
    return batch_time,data_time,losses,psnrs,accs

def print_result(epoch,train_result,test_result):
    print('Test Epoch: {0}\tLoss: {losses.avg:.4f}\tPSNR: {psnrs.avg:.4f}\tACC: {accs.avg:.4f}\tTime: {time.sum}'
        .format(epoch,losses=test_result[2],psnrs=test_result[3],accs=test_result[4],time=train_result[0]))
    return


if __name__ == "__main__":
    main()    