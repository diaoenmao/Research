import numpy as np
import torch
import os
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from util import *
sys.path.append("./CMU-MultimodalDataSDK/")
from mmdata import Dataloader, Dataset


def fetch_Multimodal_data(data_name,data_mode,batch_size):
    print('fetching data...')
    stats_name = './data/stats/stats_{}_{}.pkl'.format(data_name,data_mode)
    if(data_name=='MOSI'):
        if(os.path.exists(stats_name)):
            mean,std = load(stats_name)
        else:
            train_dataset = ModalityDataset(data_mode, 'train', download=True)
            valid_dataset = ModalityDataset(data_mode, 'valid', download=True)
            eval_dataset = torch.utils.data.ConcatDataset([train_dataset,valid_dataset])
            mean,std = get_mean_and_std(eval_dataset,'{}_{}'.format(data_name,data_mode))
        transform_train = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        train_dataset = ModalityDataset(data_mode, 'train', download=True, transform=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        valid_dataset = ModalityDataset(data_mode, 'valid', download=True, transform=None)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) 
        eval_dataset = torch.utils.data.ConcatDataset([train_dataset,valid_dataset])
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_dataset = ModalityDataset(data_mode, 'test', download=True, transform=None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) 
    return train_loader,valid_loader,eval_loader,test_loader
    
class ModalityDataset(Dataset):
    """Modality dataset."""

    def __init__(self, data_mode, mode, transform=None, target_transform=None, download=False):
        if(mode not in ['train','valid','test']):
            raise ValueError('Invalid mode')
        if(data_mode not in ['visual','audio','text','combined']):
            raise ValueError('Invalid data mode')
        self.mode = mode
        self.data_mode = data_mode
        self.root = './data/'+self.data_mode+'.pt'
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if(self.mode=='train'):
            self.x_train,_,_,self.y_train,_,_ = load(self.root)
        elif(self.mode=='valid'):
            _,self.x_valid,_,_,self.y_valid,_ = load(self.root)
        elif(self.mode=='test'):
            _,_,self.x_test,_,_,self.y_test = load(self.root)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if(self.mode=='train'):
            return self.x_train.shape[0]
        elif(self.mode=='valid'):
            return self.x_valid.shape[0]
        elif(self.mode=='test'):
            return self.x_test.shape[0]

    def __getitem__(self, idx):
        if(self.mode=='train'):
            input = torch.from_numpy(self.x_train[idx,])
            target = torch.tensor(self.y_train[idx].item())
        elif(self.mode=='valid'):
            input = torch.from_numpy(self.x_valid[idx,])
            target = torch.tensor(self.y_valid[idx].item()) 
        elif(self.mode=='test'):
            input = torch.from_numpy(self.x_test[idx,])
            target = torch.tensor(self.y_test[idx].item())
            
        if self.transform is not None: 
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target
        
    def _check_exists(self):  
        return os.path.exists(self.root)
        
    def download(self):
        if self._check_exists():
            return
        download_Multimodal_data('MOSI')
    
def pad(data, max_len):
    """A funtion for padding/truncating sequence data to a given lenght"""
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    n_rows = data.shape[0]
    dim = data.shape[1]
    if max_len >= n_rows:
        diff = max_len - n_rows
        padding = np.zeros((diff, dim))
        padded = np.concatenate((padding, data))
        return padded
    else:
        return data[-max_len:]

def download_Multimodal_data(data_name,max_len=20):
    # Download the data if not present
    data = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/'+data_name)
    embeddings = data.embeddings()
    facet = data.facet()
    covarep = data.covarep()
    sentiments = data.sentiments() # sentiment labels, real-valued. for this tutorial we'll binarize them
    train_ids = data.train() # set of video ids in the training set
    valid_ids = data.valid() # set of video ids in the valid set
    test_ids = data.test() # set of video ids in the test set

    # Merge different features and do word level feature alignment (align according to timestamps of embeddings)
    bimodal = Dataset.merge(embeddings, facet)
    trimodal = Dataset.merge(bimodal, covarep)
    dataset = trimodal.align('embeddings')

    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                train_set_ids.append((vid, sid))

    valid_set_ids = []
    for vid in valid_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                valid_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        for sid in dataset['embeddings'][vid].keys():
            if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                test_set_ids.append((vid, sid))

    # partition the training, valid and test set. all sequences will be padded/truncated to max_len steps
    # data will have shape (dataset_size, max_len, feature_dim)

    train_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
    valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
    test_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)

    train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

    train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
    valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
    test_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

    
    # the sentiment scores for 7-class classification task
    y_train = np.round(np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])).astype(np.int64)
    y_valid = np.round(np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])).astype(np.int64)
    y_test = np.round(np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])).astype(np.int64)
    ground_label = min([np.min(y_train),np.min(y_valid),np.min(y_test)])
    y_train -= ground_label
    y_valid -= ground_label
    y_test -= ground_label

    # the sentiment scores for binary classification task
    # y_train = (np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0).astype(np.int64)
    # y_valid = (np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0).astype(np.int64)
    # y_test = (np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0).astype(np.int64)
    
    # y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
    # y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
    # y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

    
    # normalize covarep and facet features, remove possible NaN values
    visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
    train_set_visual = train_set_visual / visual_max
    valid_set_visual = valid_set_visual / visual_max
    test_set_visual = test_set_visual / visual_max

    train_set_visual[train_set_visual != train_set_visual] = 0
    valid_set_visual[valid_set_visual != valid_set_visual] = 0
    test_set_visual[test_set_visual != test_set_visual] = 0

    audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
    train_set_audio = train_set_audio / audio_max
    valid_set_audio = valid_set_audio / audio_max
    test_set_audio = test_set_audio / audio_max

    train_set_audio[train_set_audio != train_set_audio] = 0
    valid_set_audio[valid_set_audio != valid_set_audio] = 0
    test_set_audio[test_set_audio != test_set_audio] = 0

    
    train_sets = [train_set_visual,train_set_audio,train_set_text]
    valid_sets = [valid_set_visual,valid_set_audio,valid_set_text]
    test_sets = [test_set_visual,test_set_audio,test_set_text]
    data_mode = ['visual','audio','text']

    for i in range(len(data_mode)):
        x_train, x_valid, x_test = train_sets[i].astype(np.float32), valid_sets[i].astype(np.float32), test_sets[i].astype(np.float32)
        save([x_train, x_valid, x_test, y_train, y_valid, y_test], './data/'+ data_mode[i] + '.pt')
    
    #early fusion: input level concatenation of features
    x_train = np.concatenate((train_set_visual, train_set_audio, train_set_text), axis=2).astype(np.float32)
    x_valid = np.concatenate((valid_set_visual, valid_set_audio, valid_set_text), axis=2).astype(np.float32)
    x_test = np.concatenate((test_set_visual, test_set_audio, test_set_text), axis=2).astype(np.float32)

    save([x_train, x_valid, x_test, y_train, y_valid, y_test], './data/combined.pt')
    return

def get_mean_and_std(dataset,data_name=''):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    num_c = list(dataloader)[0][0].size(1)
    mean = torch.zeros(num_c)
    std = torch.zeros(num_c)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(num_c):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    save([mean,std],'./data/stats/stats_{}.pkl'.format(data_name))
    return mean, std
    
