import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from util import *

MAX_BUFF_SIZE = 1024*(1024*1024) #1GB
    
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
        
 
def show_shots_transition(root,video_name):
    video_dir = '{}/videos'.format(root)
    annotation_dir = '{}/annotations'.format(root)
    cap = cv2.VideoCapture('{}/{}'.format(video_dir,video_name))
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    annotation_name = '{}.txt'.format(video_name.rsplit('.',1)[0])
    shots = annotation2label('{}/annotations/shots/{}'.format(root,annotation_name),FRAME_COUNT,0,'\t')
    scenes = annotation2label('{}/annotations/scenes/{}'.format(root,annotation_name),FRAME_COUNT,1,' ')
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
            cv2.waitKey(int(1000/FPS))
            index += 1
        else:
            break
    return
    
root = './data/SBD/RAI'
video_name = '1.mp4'
show_shots_transition(root,video_name)    