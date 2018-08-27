import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from util import *


cudnn.benchmark = True
data_name = 'MNIST'
model_dir = 'mnist'
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
if_resume = config.PARAM['if_resume']
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
    print('Training data size {}, Number of Batches {}, Test data size {}'.format(data_size,len(train_loader),len(test_dataset)))
    last_epoch = 0
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    optimizer = optim.Adam(model.parameters(),lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.CrossEntropyLoss().to(device)
    if(if_resume):
        last_epoch,model,optimizer,scheduler = resume(model,optimizer,scheduler,Experiment_TAG)
   
    best_pivot = 255
    for epoch in range(last_epoch, validated_num_epochs+1):
        scheduler.step()
        train_result = train(train_loader,model,criterion,optimizer,epoch)
        test_result = test(test_loader,model,criterion,epoch)
        print_result(epoch,train_result,test_result)
        if(save_mode>1):
            save({'epoch':epoch+1,'model_dict':model.state_dict(),'optimizer_dict':optimizer.state_dict(),'scheduler_dict': scheduler.state_dict()},'./output/model/{}_checkpoint.pkl'.format(Experiment_TAG))
            if(best_pivot>test_result[2].avg):
                best_pivot = test_result[2].avg if best_pivot>test_result[2].avg else best_pivot
                save({'epoch':epoch+1,'model_dict':model.state_dict(),'optimizer_dict':optimizer.state_dict(),'scheduler_dict': scheduler.state_dict()},'./output/model/{}_best.pkl'.format(Experiment_TAG))
                save({'test_loss':test_result[2].avg,'test_psnr':test_result[3].avg,'test_acc':test_result[4].avg,'timing':train_result[0].avg},'./output/result/{}_best.pkl.pkl'.format(Experiment_TAG))
    if(save_mode>0):
        save({'epoch':epoch+1,'model_dict':model.state_dict(),'optimizer_dict':optimizer.state_dict(),'scheduler_dict': scheduler.state_dict()},'./output/model/{}_final.pkl'.format(Experiment_TAG))  
        save({'test_loss':test_result[2].avg,'test_psnr':test_result[3].avg,'test_acc':test_result[4].avg,'timing':train_result[0].avg},'./output/result/{}_final.pkl'.format(Experiment_TAG))        
    return result
    
def train(train_loader,model,criterion,optimizer,epoch):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    accs = Meter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        data_time.update(time.time() - end)
        output = model(input)
        #loss = output[0] + 1e-3*criterion(output[2],target)
        loss = output[0]
        psnr = PSNR(output[1],input)
        acc = ACC(output[2],target,topk=(1,))
        losses.update(loss.item(), input.size(0))
        psnrs.update(psnr.item(), input.size(0))
        accs.update(acc[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % (len(train_loader)//5) == 0:
            print('Train Epoch: {}[({:.0f}%)]\tLoss: {:.4f}\tPNSR: {:.4f}\tACC: {:.4f}'.format(
                epoch, 100. * i / len(train_loader), loss.item(), psnr.item(), acc[0]))
    return batch_time,data_time,losses,psnrs,accs
  
    
def test(validation_loader,model,criterion,epoch):
    output_dir = './output/img'
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    psnrs = Meter()
    accs = Meter()
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(validation_loader):
            input = input.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            output = model(input)
            #loss = output[0] + criterion(output[2],target)
            loss = output[0]
            psnr = PSNR(output[1],input)
            acc = ACC(output[2],target,topk=(1,))
            losses.update(loss.item(), input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            accs.update(acc[0])
            batch_time.update(time.time() - end)
            end = time.time()
        if epoch % 3 == 0:
            if not os.path.exists('.{}/image_{}.png'.format(output_dir,i)):
                save_img(input,'{}/image_{}.png'.format(output_dir,i))
            save_img(output[1],'{}/image_{}_{}.png'.format(output_dir,i,epoch))
    return batch_time,data_time,losses,psnrs,accs

def print_result(epoch,train_result,test_result):
    print('Test Epoch: {0}\tLoss: {losses.avg:.4f}\tPSNR: {psnrs.avg:.4f}\tACC: {accs.avg:.4f}\tTime: {time.sum}'
        .format(epoch,losses=test_result[2],psnrs=test_result[3],accs=test_result[4],time=train_result[0]))
    return

def resume(model,optimizer,scheduler,Experiment_TAG):
    if(os.path.exists('./output/model/{}_checkpoint.pkl'.format(Experiment_TAG))):
        checkpoint = load('./output/model/{}_checkpoint.pkl'.format(Experiment_TAG))
        last_epoch = checkpoint['last_epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
    else:
        last_epoch = 0
    return last_epoch,model,optimizer,scheduler
        
if __name__ == "__main__":
    main()    