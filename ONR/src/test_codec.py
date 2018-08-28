import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from data import *
from util import *
from codec import *

cudnn.benchmark = True
data_name = 'MNIST'
model_dir = 'mnist'
model_name = 'RCAE'
TAG = data_name+'_'+model_name
special_TAG = 'handtuned'
config.init()
patch_shape = config.PARAM['patch_shape']
batch_size = config.PARAM['batch_size']
data_size = config.PARAM['data_size']
device = torch.device(config.PARAM['device'])    
init_seed = 0
num_Experiments = 1
seeds = list(range(init_seed,init_seed+num_Experiments))
num_iter = 16

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
    
    _,test_dataset = fetch_dataset(data_name=data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    best = load('./output/model/{}_best_{}.pkl'.format(Experiment_TAG,special_TAG))
    rate_distortion = np.zeros((2,num_iter))
    for i in range(1,num_iter+1):
        model = eval('models.{}.{}(num_iter={}).to(device)'.format(model_dir,model_name,i))
        last_epoch = best['epoch']
        model.load_state_dict(best['model_dict'])
        criterion = nn.CrossEntropyLoss().to(device)
        test_result = test(test_loader,model,criterion)
        print_result(test_result)
        rate_distortion[0,i-1],rate_distortion[1,i-1] = test_result[3].avg,test_result[4].avg
        if(i==num_iter):
            acc = test_result[5].avg
    save({'rate_distortion':rate_distortion,'acc':acc},'./output/result/{}_{}.pkl'.format(Experiment_TAG,special_TAG))  
    return
    
def test(validation_loader,model,criterion):
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    bpps = Meter()
    psnrs = Meter()
    accs = Meter()
    model.eval()
    codec = Codec(model,device)
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(validation_loader):
            input = input.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            codes = model.encode(input)
            decoded_output = model.decode(codes,input.size(0))            
            bpp = BPP(codec.entropy_encode(codes),input.numel())
            psnr = PSNR(decoded_output,input)
            if(model.num_iter==16):
                output = model.net(torch.cat(codes,1))
                loss = model.compression_loss_fn(decoded_output,input) + 1e-3*criterion(output,target)
                acc = ACC(output,target,topk=(1,))
                losses.update(loss.item(), input.size(0))
                accs.update(acc[0])
            bpps.update(bpp, input.size(0))
            psnrs.update(psnr.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        if not os.path.exists('./output/img/image_{}.png'.format(i)):
            save_img(input,'./output/img/image_{}.png'.format(i))
        save_img(decoded_output,'./output/img/image_{}_{}.png'.format(i,model.num_iter))
    return batch_time,data_time,losses,bpps,psnrs,accs

def print_result(test_result):
    print('Test: Loss: {losses.avg:.4f}\tBPP: {bpps.avg:.4f}\tPSNR: {psnrs.avg:.4f}\tACC: {accs.avg:.4f}'
        .format(losses=test_result[2],bpps=test_result[3],psnrs=test_result[4],accs=test_result[5]))
    return
    
if __name__ == "__main__":
    main()