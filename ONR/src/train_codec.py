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


cudnn.benchmark = False
data_name = 'ImageNet'
model_dir = 'imagenet'
model_name = 'RCAE'
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
    
    train_dataset,_ = fetch_dataset(data_name=data_name)
    _,test_dataset = fetch_dataset(data_name='Kodak')
    validated_num_epochs = max_num_epochs
    
    train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,batch_size=1,num_fold=0,radomGen=randomGen)
    print('Training data size {}, Test data size {}'.format(len(train_loader),len(test_dataset)))
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
    result = {'test_loss':test_loss,'test_psnr':test_psnr,'timing':timing}
    if(save_mode>0):
        save({'model_dict':mw.model.state_dict(),'optimizer_dict':mw.optimizer.state_dict()},'./output/model/{}.pkl'.format(Experiment_TAG))  
        save(result,'./output/result/{}.pkl'.format(Experiment_TAG))        
    return result
    
def train(train_loader,mw,epoch):
    batch_time = Meter()
    data_time = Meter()
    patch_size = Meter()
    losses = Meter()
    psnrs = Meter()
    accs = Meter()
    mw.model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        data_time.update(time.time() - end)
        input_size.update(input.size(0))
        output = mw.model(input)
        loss = mw.loss(output,target)
        psnr = PSNR(output[0],input)
        #acc = mw.acc
        losses.update(loss.item(), input_size.size(0))
        psnrs.update(psnr.item(), input_size(0))
        accs.update(acc.item())
        mw.optimizer.zero_grad()
        loss.backward()
        mw.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
    if i % (len(train_loader)//5) == 0:
        print('Train Epoch: {}[({:.0f}%)]\tLoss: {:.6f}\tPNSR: {:.3f}'.format(
            epoch, 100. * i / len(train_loader), loss.item(), psnr.item()))
    return batch_time,data_time,patch_size,losses,psnrs
  
    
def test(validation_loader,mw,epoch):
    output_dir = './output/img'
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    mw.model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(validation_loader):
            input = input.to(device)
            data_time.update(time.time() - end)
            output = mw.model(input)
            loss = mw.loss(output,target)
            psnr = PSNR(output[0],input)
            losses.update(loss.item(), input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        if epoch % 3 == 0:
            nrow = int(np.ceil(float(input.size(3))/patch_shape[1]))
            if not os.path.exists('.{}/image_{}.png'.format(output_dir,i)):
                save_img(patches,'{}/image_{}.png'.format(output_dir,i),nrow)
            save_img(output,'{}/image_{}_{}.png'.format(output_dir,i,epoch),nrow)
    return batch_time,data_time,losses,psnrs


if __name__ == "__main__":
    main()    