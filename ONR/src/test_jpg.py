import torch
import config
import time
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from data import *
from util import *
from codec import *

cudnn.benchmark = True
data_name = 'MNIST'
TAG = data_name
special_TAG = 'jpeg'
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
    
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.Lambda(lambda x: PIL_to_CV2(x))])  
    _,test_dataset = fetch_dataset(data_name=data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True)
    qualities = list(range(5,101,5))
    rate_distortion = np.zeros((2,len(qualities)))
    for i in range(len(qualities)):
        test_result = test(test_loader,qualities[i])
        print_result(test_result)
        rate_distortion[0,i],rate_distortion[1,i] = test_result[2].avg,test_result[3].avg
    save({'rate_distortion':rate_distortion},'./output/result/{}_{}.pkl'.format(Experiment_TAG,special_TAG))  
    return
    
def test(validation_loader,quality):
    batch_time = Meter()
    data_time = Meter()
    bpps = Meter()
    psnrs = Meter()
    end = time.time()
    for i, (input, _) in enumerate(validation_loader):
        input = input.to(device)
        npy_input = input.squeeze().cpu().numpy()*255
        data_time.update(time.time() - end)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        retval, codes = cv2.imencode('.jpg', npy_input.astype(np.uint8), encode_param)
        decoded_output = cv2.imdecode(codes,0)
        decoded_output = torch.from_numpy(decoded_output.astype(np.float32)).to(device)
        bpp = BPP(codes,input.numel())
        psnr = PSNR(decoded_output,torch.from_numpy(npy_input).to(device),max=255.0)
        bpps.update(bpp, input.size(0))
        psnrs.update(psnr.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    if not os.path.exists('./output/img/image_{}.png'.format(i)):
        save_img(input,'./output/img/image_{}.png'.format(i))
    save_img(decoded_output/255,'./output/img/image_{}_{}.png'.format(i,quality))
    return batch_time,data_time,bpps,psnrs
   
def print_result(test_result):
    print('Test: BPP: {bpps.avg:.4f}\tPSNR: {psnrs.avg:.4f}'
        .format(bpps=test_result[2],psnrs=test_result[3]))
    return
if __name__ == "__main__":
    main()