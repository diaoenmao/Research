import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from util import *

MAX_BUFF_SIZE = 1024*(1024*1024) #1GB

def video2arr(path):
    cap = cv2.VideoCapture(path)
    inputfile_dir = path.rsplit('.',1)[0]
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    BUFF_COUNT = int(MAX_BUFF_SIZE/(3*FRAME_HEIGHT*FRAME_WIDTH))
    buf = np.empty((BUFF_COUNT,FRAME_HEIGHT,FRAME_WIDTH,3), dtype=np.uint8)
    FILE_FRAME_COUNT = []
    inputfile_idx = 0
    input_idx = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow('frame',frame)
            cv2.waitKey(int(1000/FPS))
            buf[input_idx,] = frame
            if(input_idx==BUFF_COUNT-1):
                print(inputfile_idx)
                save(buf,'{}/{}'.format(inputfile_dir,inputfile_idx),mode='numpy')
                buf = np.empty((BUFF_COUNT,FRAME_HEIGHT,FRAME_WIDTH,3), dtype=np.uint8)
                inputfile_idx += 1
                input_idx = 0
                FILE_FRAME_COUNT.append(BUFF_COUNT)
            else:
                input_idx += 1
        else:
            if(input_idx!=0):
                print(inputfile_idx)
                save(buf[:input_idx,],'{}/{}'.format(inputfile_dir,inputfile_idx),mode='numpy')
                FILE_FRAME_COUNT.append(input_idx)
                FILE_FRAME_COUNT = np.array(FILE_FRAME_COUNT,dtype=np.int64)
                META = {'FILE_FRAME_COUNT':FILE_FRAME_COUNT,'FRAME_WIDTH':FRAME_WIDTH,'FRAME_HEIGHT':FRAME_HEIGHT,'FPS':FPS}
                save(META, '{}/META.pkl'.format(inputfile_dir))
            break
    cap.release()
    return

def arr2video(input_dir):
    META = load('{}/META.pkl'.format(input_dir))  
    for i in range(META['FILE_FRAME_COUNT'].shape[0]):
        s = time.time()
        cur_input = load('{}/{}.npy'.format(input_dir,i),mode='numpy')
        e = time.time()
        print(e-s)
        for j in range(cur_input.shape[0]):
            frame = cur_input[j,]
            cv2.imshow('frame',frame)
            cv2.waitKey(int(1000/META['FPS']))
    return
    
def annotation2label(path,FRAME_COUNT,start=0,delimiter=','):
    annotation = np.genfromtxt(path,dtype=np.int64,delimiter=delimiter)-start
    label = torch.zeros(FRAME_COUNT,dtype=torch.long)
    for i in range(annotation.shape[0]-1):
        label[annotation[i,1]:annotation[i+1,0]+1] = 1
    return label

def filenames_in(dir):
    filenames_ext = os.listdir(dir)
    filenames = [filename.rsplit('.',1)[0] for filename in filenames_ext]
    filenames.sort()
    return filenames
   
class RAIDataset(Dataset):
    def __init__(self,root,video_name,transform=None,target_transform=None):
        self.root = root
        self.video_name = video_name
        self.name = video_name.rsplit('.',1)[0]
        self.video_dir = '{}/videos'.format(self.root)
        self.annotation_dir = '{}/annotations'.format(self.root)
        self.input_dir = '{}/{}'.format(self.video_dir,self.name)
        self.inputfile_names = filenames_in(self.input_dir)
        self.META = load('{}/META.pkl'.format(self.input_dir))        
        self.target_scenes = annotation2label('{}/scenes/{}.txt'.format(self.annotation_dir,self.name),np.sum(self.META['FILE_FRAME_COUNT']).item(),start=1,delimiter=' ')
        self.target_shots = annotation2label('{}/shots/{}.txt'.format(self.annotation_dir,self.name),np.sum(self.META['FILE_FRAME_COUNT']).item(),start=0,delimiter='\t')
        self.inputfile_idx = 0
        self.inputfile = load('{}/{}.npy'.format(self.input_dir,self.inputfile_names[self.inputfile_idx]),mode='numpy')
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        cumsum_count = np.cumsum(self.META['FILE_FRAME_COUNT'])
        cur_inputfile_idx = np.nonzero(cumsum_count>index)[0][0]
        input_idx = index-(cumsum_count[cur_inputfile_idx]-self.META['FILE_FRAME_COUNT'][cur_inputfile_idx])
        if(cur_inputfile_idx!=self.inputfile_idx):
            self.inputfile = load('{}/{}.npy'.format(self.input_dir,self.inputfile_names[cur_inputfile_idx]),mode='numpy')
            self.inputfile_idx = cur_inputfile_idx
        target_scene = self.target_scenes[index]
        target_shot = self.target_shots[index]
        input = self.inputfile[input_idx,]
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (input,target_scene,target_shot)

    def __len__(self):
        return np.sum(self.META['FILE_FRAME_COUNT']) 
        
def show_shots_transition_1(root,video_name):
    video_dir = '{}/videos'.format(root)
    annotation_dir = '{}/annotations'.format(root)
    name = video_name.rsplit('.',1)[0]
    input_dir = '{}/{}'.format(video_dir,name)
    if(not os.path.exists('{}/META.pkl'.format(input_dir))):
        video2arr('{}/{}'.format(video_dir,video_name))
    arr2video(input_dir)
    exit()
    dataset = RAIDataset(root,video_name)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (input,target_scene,target_shot) in enumerate(data_loader):
        print(i,target_shot.item())
        input = input.squeeze().numpy()
        cv2.imshow('video',input)
        if(target_shot.item()==1):
            cv2.imshow('annotation',np.zeros((320,320)))
        else:
            cv2.imshow('annotation',np.ones((320,320)))
        cv2.waitKey(int(1000/dataset.META['FPS']))
    return
 
def show_shots_transition_2(root,video_name):
    video_dir = '{}/videos'.format(root)
    annotation_dir = '{}/annotations'.format(root)
    cap = cv2.VideoCapture('{}/{}'.format(video_dir,video_name))
    dataset = RAIDataset(root,video_name)
    shots = dataset.target_shots
    scenes = dataset.target_scenes
    index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            print(index,shots[index].item(),scenes[index].item())
            cv2.imshow('video',frame)
            if(shots[index].item()==1):
                cv2.imshow('shots',np.zeros((320,320)))
            else:
                cv2.imshow('shots',np.ones((320,320)))
            if(scenes[index].item()==1):
                cv2.imshow('scenes',np.zeros((320,320)))
            else:
                cv2.imshow('scenes',np.ones((320,320)))
            cv2.waitKey(int(1000/dataset.META['FPS']))
            index += 1
        else:
            break
    return
    
root = './data/RAI'
video_name = '0.mp4'
show_shots_transition_2(root,video_name)    