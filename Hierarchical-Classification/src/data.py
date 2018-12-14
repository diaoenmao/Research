import numpy as np
import torch
import os
import torch.utils.data as data_utils
import tarfile
import config
import datasets
import datasets.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils import *


config.init()

seed = 1234
if_dist = False
world_size = config.PARAM['world_size']
num_workers = config.PARAM['num_workers']
branch = config.PARAM['branch']
device = config.PARAM['device']

def fetch_dataset(data_name):
    print('fetching data {}...'.format(data_name))
    if(data_name=='MNIST'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/test/'.format(data_name)
        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])            
        train_dataset = datasets.MNIST(root=train_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=test_dir, train=False, download=True, transform=transform)

    elif(data_name=='EMNIST' or data_name=='EMNIST_byclass' or data_name=='EMNIST_bymerge' or
        data_name=='EMNIST_balanced' or data_name=='EMNIST_letters' or data_name=='EMNIST_digits' or data_name=='EMNIST_mnist'):
        train_dir = './data/{}/train/'.format(data_name.split('_')[0])
        test_dir = './data/{}/test/'.format(data_name.split('_')[0])
        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])
        split = 'balanced' if len(data_name.split('_')) == 1 else data_name.split('_')[1]
        train_dataset = datasets.EMNIST(root=train_dir, split=split, branch=branch, train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root=test_dir, split=split, branch=branch, train=False, download=True, transform=transform)

    elif(data_name=='FashionMNIST'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/test/'.format(data_name)
        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])            
        train_dataset = datasets.FashionMNIST(root=train_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=test_dir, train=False, download=True, transform=transform)
        
    elif(data_name=='CIFAR10'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/validation/'.format(data_name)
        train_dataset = datasets.CIFAR10(train_dir, train=True, transform=transforms.ToTensor(), download=True)
        mean, std = make_stats(data_name,train_dataset)
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train_dataset.transform=train_transform
        test_dataset = datasets.CIFAR10(test_dir, train=False, transform=test_transform, download=True)

    elif(data_name=='CIFAR100'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/validation/'.format(data_name)
        train_dataset = datasets.CIFAR100(train_dir, branch=branch, train=True, transform=transforms.ToTensor(), download=True)
        mean, std = make_stats(data_name,train_dataset)
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train_dataset.transform=train_transform
        test_dataset = datasets.CIFAR100(test_dir, branch=branch, train=False, transform=test_transform, download=True)
        
    elif(data_name=='SVHN'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/validation/'.format(data_name)
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.SVHN(train_dir, split='train', transform=transform, download=True)
        test_dataset = datasets.SVHN(test_dir, split='test', transform=transform, download=True)
        
    elif(data_name=='ImageNet'):
        train_dir = './data/{}/train/'.format(data_name)
        test_dir = './data/{}/validation/'.format(data_name)
        train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        mean, std = make_stats(data_name,train_dataset)
        transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train_dataset.transform=transform
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    elif(data_name=='WheatImage'or data_name=='WheatImage_binary' or data_name=='WheatImage_six'):
        train_dir = './data/{}/train/'.format(data_name.split('_')[0])
        test_dir = './data/{}/validation/'.format(data_name.split('_')[0])
        split = 'six' if len(data_name.split('_')) == 1 else data_name.split('_')[1]
        train_dataset = datasets.WheatImage(train_dir, split=split, transform=transforms.ToTensor())
        mean, std = make_stats(data_name,train_dataset)
        train_transform = transforms.Compose([transforms.Resize((256,352)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.Resize((256,352)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train_dataset.transform=train_transform
        test_dataset = datasets.WheatImage(test_dir, split=split, transform=test_transform)
            
    elif(data_name=='CocoDetection'):
        train_dir = './data/Coco/train2017/'
        train_ann = './data/Coco/annotations/instances_train2017.json'
        test_dir = './data/Coco/val2017/'
        test_ann = './data/Coco/annotations/instances_val2017.json'
        transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor()])
        train_dataset = datasets.CocoDetection(
            train_dir, train_ann, transform=transform)
        test_dataset = datasets.CocoDetection(
            test_dir, test_ann, transform=transform)

    elif(data_name=='CocoCaptions'):
        train_dir = './data/Coco/train2017/'
        train_ann = './data/Coco/annotations/captions_train2017.json'
        test_dir = './data/Coco/val2017/'
        test_ann = './data/Coco/annotations/captions_val2017.json'
        transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor()])
        train_dataset = datasets.CocoCaptions(
            train_dir, train_ann, transform=transform)
        test_dataset = datasets.CocoCaptions(
            test_dir, test_ann, transform=transform)
            
    elif(data_name=='VOCDetection'):
        train_dir = './data/VOC/VOCdevkit/'
        test_dir = './data/VOC/VOCdevkit/'
        transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor()])
        train_dataset = datasets.VOCDetection(
            train_dir, 'trainval', transform=transform)
        test_dataset = datasets.VOCDetection(
            test_dir, 'test', transform=transform)

    elif(data_name=='VOCSegmentation'):
        train_dir = './data/VOC/VOCdevkit/'
        test_dir = './data/VOC/VOCdevkit/'
        transform = transforms.Compose([transforms.Resize((256,256)),
                                        transforms.ToTensor()])
        train_dataset = datasets.VOCSegmentation(train_dir, 'trainval', transform=transform)
        test_dataset = datasets.VOCSegmentation(test_dir, 'test', transform=transform)
        
    elif(data_name =='Kodak'):
        train_dataset = None
        transform = transforms.Compose([transforms.ToTensor()])
        test_dir = './data/{}/'.format(data_name)
        test_dataset = datasets.ImageFolder(
            test_dir, transform)
            
    elif(data_name =='UCID'):
        train_dataset = None
        transform = transforms.Compose([transforms.ToTensor()])
        test_dir = './data/{}/'.format(data_name)
        test_dataset = datasets.ImageFolder(
            test_dir, transform)
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return train_dataset,test_dataset

def input_collate(batch):
    if(isinstance(batch[0], dict)):
        output = {key: [] for key in batch[0].keys()}
        cat_mode = {key: 0 for key in batch[0].keys()}
        prev_b = batch[0]
        for b in batch:
            for key in b:
                output[key].append(b[key])
                if(prev_b[key].size() != b[key].size() or key=='bbox' or b[key] is None):
                    cat_mode[key] = 1
            prev_b = b
        for key in batch[0]:
            if(cat_mode[key]==0):
                output[key] = default_collate(output[key])
            elif(cat_mode[key]==1):
                pass
            else:
                raise ValueError('cat mode not supported')
        return output
    else:
        return default_collate(batch)
   
def split_dataset(train_dataset,test_dataset,data_size,batch_size,radomGen,collate_fn=input_collate):
    indices = list(range(len(train_dataset)))
    data_idx = radomGen.choice(indices, size=data_size, replace=False)
    if(isinstance(batch_size, int)):
        batch_size = [batch_size, batch_size]
    for i in range(2):
        if(batch_size[i]==0):
            batch_size[i] = data_size
        else:
            batch_size[i] = batch_size[i]*world_size
    train_batch_size,test_batch_size = batch_size
    train_dataset = torch.utils.data.Subset(train_dataset, data_idx)
    train_sampler = DistributedSampler(train_dataset) if (world_size > 1 and if_dist) else None           
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                shuffle=(train_sampler is None), batch_size=train_batch_size, pin_memory=True, sampler=train_sampler, num_workers=num_workers*world_size, collate_fn=collate_fn)    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                batch_size=test_batch_size, pin_memory=True, num_workers=num_workers*world_size, collate_fn=collate_fn)
    return train_loader,test_loader
    
def split_dataset_cross_validation(train_dataset,test_dataset,data_size,batch_size,num_fold,radomGen,p=0.8):
    indices = list(range(len(train_dataset)))
    data_idx = radomGen.choice(indices, size=data_size, replace=False)
    if(batch_size==0):
        batch_size = len(train_idx)
    else:
        batch_size = batch_size*world_size
    if(num_fold==1):
        train_idx = radomGen.choice(data_idx, size=int(data_size*p), replace=False)
        sub_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        train_sampler = DistributedSampler(sub_train_dataset) if (world_size > 1 and if_dist) else None
        train_loader = [torch.utils.data.DataLoader(dataset=sub_train_dataset, 
                    shuffle=(train_sampler is None), batch_size=batch_size, pin_memory=True, sampler=train_sampler, num_workers=num_workers*world_size)]   
        validation_idx = list(set(data_idx) - set(train_idx))
        validation_dataset = torch.utils.data.Subset(train_dataset, validation_idx)
        validation_sampler = DistributedSampler(validation_dataset) if (world_size > 1 and if_dist) else None
        validation_loader = [torch.utils.data.DataLoader(dataset=validation_dataset, 
                    batch_size=batch_size, pin_memory=True, sampler=validation_sampler, num_workers=num_workers*world_size)]
    elif(num_fold>1 and num_fold<=len(indices)):
        splitted_idx = np.array_split(data_idx, num_fold)
        train_loader = []
        validation_loader = []
        for i in range(num_fold):
            validation_idx = splitted_idx[i]
            train_idx = list(set(data_idx) - set(validation_idx))
            cur_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            cur_train_sampler = DistributedSampler(cur_train_dataset) if (world_size > 1 and if_dist) else None
            train_loader.append(torch.utils.data.DataLoader(dataset=cur_train_dataset, 
                shuffle=(cur_train_sampler is None), batch_size=batch_size, pin_memory=True, sampler=cur_train_sampler, num_workers=num_workers*world_size)) 
            validation_dataset = torch.utils.data.Subset(train_dataset, validation_idx)
            validation_sampler = DistributedSampler(validation_dataset) if (world_size > 1 and if_dist) else None
            validation_loader.append(torch.utils.data.DataLoader(dataset=train_dataset, 
                batch_size=batch_size, pin_memory=True, sampler=validation_sampler, num_workers=num_workers*world_size))
    else:
        error("Invalid number of fold")
        exit()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                batch_size=batch_size, pin_memory=True, num_workers=num_workers*world_size)
    return train_loader,validation_loader,test_loader
    
def fetch_dataset_synth(input_feature,output_feature,high_dim=None,cov_mode='base',noise_sigma=np.sqrt(0.1),randomGen = np.random.RandomState(seed)):
    print('fetching data...')
    data_size = 50000
    test_size = 10000
    V = make_cov_mat(input_feature,cov_mode)
    X = randomGen.multivariate_normal(np.zeros(input_feature),V,data_size+test_size)
    if(high_dim is None):
            beta = randomGen.randn(input_feature,output_feature)           
    else:
        if(high_dim>=input_feature):
            raise ValueError('invalid high dimension')
        valid_beta = randomGen.randn(high_dim,output_feature)
        empty_beta = np.zeros((input_feature-high_dim,output_feature))
        beta = np.vstack((valid_beta,empty_beta))
    mu = np.matmul(X,beta)
    eps = noise_sigma*randomGen.randn(*mu.shape)
    if(output_feature==1):
        y = mu + eps
    elif(output_feature>1):      
        p = softmax(mu + eps)
        y = []
        for i in range(X.shape[0]):
            sample = randomGen.multinomial(1,p[i,])
            y.append(np.where(sample==1)[0][0])
        y = np.array(y)
    else:
        raise ValueError('invalid dimension')
    print('data ready')
    X,y = X.astype(np.float32),y.astype(np.int64)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[:data_size,:]), torch.from_numpy(y[:data_size]))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[data_size:,:]), torch.from_numpy(y[data_size:]))
    return train_dataset,test_dataset
    
def make_cov_mat(dim,mode,zo=0.5):
    if(mode=='base'):
        V = np.eye(dim)
    elif(mode=='corr'):
        V = np.full((dim, dim), zo)
        V = V + (1-zo)*np.eye(dim)
    elif(mode=='decay_corr'):
        indices = np.arange(dim)
        valid_indices = [indices,indices]
        mesh_indices = np.meshgrid(*valid_indices, sparse=False, indexing='ij')
        exponent = np.abs(mesh_indices[0]-mesh_indices[1])
        V = np.power(zo,exponent)
    else:
        raise ValueError('invalid covariance mode')
    return V

def make_stats(data_name,dataset=None,n_channel=3):
    if(os.path.exists('./data/stats/{}.pkl'.format(data_name))):
        stats = load('./data/stats/{}.pkl'.format(data_name))
        mean,std = stats['mean'],stats['std']
    elif(dataset is not None):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers*world_size)
        mean = torch.zeros(n_channel)
        std = torch.zeros(n_channel)
        print('Computing mean and std...')
        for input in data_loader:
            img = input['img']
            for i in range(n_channel):
                mean[i] += img[:,i,].mean()
                std[i] += img[:,i,].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        save({'mean':mean,'std':std},'./data/stats/{}.pkl'.format(data_name))
    else:
        raise ValueError('Please provide dataset for making stats')
    print('Data Mean: {}, Std: {}'.format(mean,std))
    return mean, std
    
def unzip(path,mode='zip'):
    filenames = filenames_in(path,mode)
    for filename in filenames:
        print('Unzipping {}'.format(filename),end='')
        tar = tarfile.open('{}/{}.{}'.format(path,filename,mode))
        tar.extractall(path='{}/{}'.format(path,filename))
        tar.close()
        print('Done')
    return
            
def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches

def reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img
    
    
    
    
    