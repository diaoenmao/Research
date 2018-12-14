import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
import torch.optim as optim
import os
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from utils import *

cudnn.benchmark = True
config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
init_seed = 0
seeds = list(range(init_seed,init_seed+num_Experiments))


def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment(seeds[i])
    return

def runExperiment(seed):
    print(config.PARAM)
    resume_model_TAG = '{}_{}_{}'.format(seed,model_data_name,model_name) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seed,model_data_name,model_name,resume_TAG)
    model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    _,test_dataset = fetch_dataset(data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*world_size, pin_memory=True, num_workers=num_workers*world_size, collate_fn = input_collate)
    best = load('./output/model/{}_best.pkl'.format(resume_model_TAG))
    last_epoch = best['epoch']
    print('Test from {}'.format(last_epoch))
    model = eval('models.{}.{}(classes_size=test_dataset.classes_size)'.format(model_dir,model_name)) 
    model.load_state_dict(best['model_dict'])
    test_meter_panel = test(test_loader,model,last_epoch,model_TAG)
    print_result(last_epoch,test_meter_panel)
    save({'config':config.PARAM,'epoch':last_epoch,'loss':test_meter_panel.panel['loss'],'acc':test_meter_panel.panel['acc']},'./output/result/{}.pkl'.format(model_TAG))  
    return
            
def test(validation_loader,model,epoch,model_TAG):
    meter_panel = Meter_Panel(meter_names)
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):          
            input = input_to_device(input,device)
            protocol = set_protocol(input)  
            output = model(input,protocol)
            output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
            evaluation = meter_panel.eval(input,output,protocol)
            batch_time = time.time() - end
            meter_panel.update(evaluation,input['img'].size(0))
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
    return meter_panel
     
def set_protocol(input):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param']
    protocol['num_iter'] = config.PARAM['num_iter']
    protocol['depth'] = config.PARAM['max_depth']
    protocol['img_shape'] = (input['img'].size(2),input['img'].size(3))
    protocol['patch_shape'] = config.PARAM['patch_shape']
    protocol['step'] = config.PARAM['step']
    protocol['jump_rate'] = config.PARAM['jump_rate']
    if(input['img'].size(1)==1):
        protocol['mode'] = 'L'
    elif(input['img'].size(1)==3):
        protocol['mode'] = 'RGB'
    else:
        raise ValueError('Wrong number of channel')
    return protocol  
        
def print_result(epoch,test_meter_panel):
    print('Test Epoch: {}{}'.format(epoch,test_meter_panel.summary(['loss','acc'])))
    return

def resume(model,optimizer,scheduler,resume_model_TAG):
    if(os.path.exists('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))):
        checkpoint = load('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        print('Resume from {}'.format(last_epoch))
    else:
        last_epoch = 0
        print('Not found existing model, and start from epoch {}'.format(last_epoch))
    return last_epoch,model,optimizer,scheduler
    
if __name__ == "__main__":
    main()    