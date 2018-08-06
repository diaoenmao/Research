import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from util import *
from modelWrapper import *
from codec import *

cudnn.benchmark = True
data_name = 'ImageNet'
model_dir = 'imagenet'
model_name = 'cae'
milestones = [50,150,250]
data_size = 50000
metric = ['PSNR','bpp']
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
    
    _,test_dataset = fetch_dataset(data_name=data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=2)
    model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
    model = load_model(model,'./output/model/{}.pkl'.format(Experiment_TAG))
    codec = Codec(model)
    test_result = test(test_loader,codec)
    print('PSNR {psnr.avg:.4f}\t'
        'BPP {bpp.avg:.4f}\t'
        .format(psnr=test_result[2],bpp=test_result[3]))
    return result

    
def test(validation_loader,codec):
    batch_time = Meter()
    data_time = Meter()
    psnrs = Meter()
    bpps = Meter()
    with torch.no_grad():
        end = time.time()
        for _, (input, _) in enumerate(validation_loader):
            input = input.to(device)
            input = extract_patches_2D(input,input_feature)
            data_time.update(time.time() - end)
            code = codec.encode(input)
            decoded_input = codec.decode(code)
            psnr = PSNR(input,decoded_input)
            bpp = BPP(input,code)
            psnrs.update(psnr.item(), input.size(0))
            bpps.update(bpp.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()              
    return batch_time,data_time,psnrs,bpps
