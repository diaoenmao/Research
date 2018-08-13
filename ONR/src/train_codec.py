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
model_name = 'cae'
TAG = data_name+'_'+model_name
config.init()
milestones = config.PARAM['milestones']
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
    _,test_dataset = fetch_dataset(data_name='Kodak')
    #validated_num_epochs = validate_num_epochs(train_dataset)
    validated_num_epochs = max_num_epochs
    
    train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,batch_size=1,num_fold=0,radomGen=randomGen)
    print('Training data size {}, Test data size {}'.format(len(train_loader),len(test_dataset)))
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    summary(model, input_size=(1, 128, 128))
    criterion_MSE = nn.MSELoss().to(device)
    mw = modelWrapper(model,config.PARAM['optimizer_name'])
    mw.set_optimizer_param(config.PARAM['optimizer_param'])
    mw.set_criterion(criterion_MSE)
    mw.set_optimizer()
        
    scheduler = MultiStepLR(mw.optimizer, milestones=milestones, gamma=0.1)
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

def validate_num_epochs(train_dataset):
    train_loader,validation_loader = split_dataset(train_dataset,None,data_size,1,1,radomGen=randomGen)
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
    return validated_num_epochs
    
def train(train_loader,mw,epoch):
    batch_time = Meter()
    data_time = Meter()
    patch_size = Meter()
    losses = Meter()
    psnrs = Meter()
    mw.model.train()
    end = time.time()
    for i, (input, _) in enumerate(train_loader):
        patches = extract_patches_2D(input,patch_shape)
        patch_dataset = torch.utils.data.TensorDataset(patches)
        patch_loader = torch.utils.data.DataLoader(dataset=patch_dataset, batch_size=batch_size, pin_memory=True)
        for j, (patch,) in enumerate(patch_loader):
            patch = patch.to(device)
            data_time.update(time.time() - end)
            patch_size.update(patch.size(0))
            output = mw.model(patch)
            loss = mw.loss(output,patch)
            psnr = PSNR(patch,output)
            losses.update(loss.item(), patch.size(0))
            psnrs.update(psnr.item(), patch.size(0))
            mw.optimizer.zero_grad()
            loss.backward()
            mw.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
        if i % (len(train_loader)/5) == 0:
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
        for i, (input, _) in enumerate(validation_loader):
            patches = extract_patches_2D(input,patch_shape)
            patch_dataset = torch.utils.data.TensorDataset(patches)
            patch_loader = torch.utils.data.DataLoader(dataset=patch_dataset, batch_size=batch_size, pin_memory=True)
            for j, (patch,) in enumerate(patch_loader):
                patch = patch.to(device)
                data_time.update(time.time() - end)
                output = mw.model(patch)
                loss = mw.loss(output,patch)
                psnr = PSNR(patch,output)
                losses.update(loss.item(), patch.size(0))
                psnrs.update(psnr.item(), patch.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        if epoch % 3 == 0:
            nrow = int(np.ceil(float(input.size(3))/patch_shape[1]))
            if not os.path.exists('.{}/image.png'.format(output_dir)):
                save_img(patches,nrow,'{}/image.png'.format(output_dir))
            save_img(output,nrow,'{}/image_{}.png'.format(output_dir,epoch))
    return batch_time,data_time,losses,psnrs


if __name__ == "__main__":
    main()    