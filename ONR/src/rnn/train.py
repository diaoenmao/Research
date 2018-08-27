import time
import os
import argparse
import config
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import torch.backends.cudnn as cudnn
from data import *
from util import *
from modelWrapper import *

cudnn.benchmark = True
config.init()
device = torch.device(config.PARAM['device'])

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=32, help='batch size')
# parser.add_argument(
    # '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()
data_size = 60000
## load 32x32 patches from images

seed = 0
randomGen = np.random.RandomState(seed)
train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])
train_dataset,test_dataset = fetch_dataset(data_name='MNIST')
train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,batch_size=32,num_fold=0,radomGen=randomGen)

print('total images: {}; total batches: {}'.format(
    data_size, len(train_loader)))

## load networks on GPU
# import network
import RCAE
# encoder = network.EncoderCell().cuda()
# binarizer = network.Binarizer().cuda()
# decoder = network.DecoderCell().cuda()

model = RCAE.RCAE().cuda()
mw = modelWrapper(model,config.PARAM['optimizer_name'])
mw.set_optimizer_param(config.PARAM['optimizer_param'])
criterion = nn.CrossEntropyLoss().to(device)
mw.set_criterion(criterion)
mw.set_optimizer()
# solver = optim.Adam(
    # [
        # {
            # 'params': encoder.parameters()
        # },
        # {
            # 'params': binarizer.parameters()
        # },
        # {
            # 'params': decoder.parameters()
        # },
    # ],
    # lr=args.lr)
optimizer = optim.Adam(model.parameters(),lr=args.lr)

# def resume(epoch=None):
    # if epoch is None:
        # s = 'iter'
        # epoch = 0
    # else:
        # s = 'epoch'

    # encoder.load_state_dict(
        # torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    # binarizer.load_state_dict(
        # torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    # decoder.load_state_dict(
        # torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


# def save(index, epoch=True):
    # if not os.path.exists('checkpoint'):
        # os.mkdir('checkpoint')

    # if epoch:
        # s = 'epoch'
    # else:
        # s = 'iter'

    # torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(
        # s, index))

    # torch.save(binarizer.state_dict(),
               # 'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    # torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(
        # s, index))


# resume()
output_dir = './output/img'
scheduler = LS.MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0
# if args.checkpoint:
    # resume(args.checkpoint)
    # last_epoch = args.checkpoint
    # scheduler.last_epoch = last_epoch - 1

for epoch in range(last_epoch, args.max_epochs + 1):

    scheduler.step()

    for i, (data, target) in enumerate(train_loader):
        print(i)
        batch_t0 = time.time()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        bp_t0 = time.time()
        compression_loss,image,output = model(data)
        bp_t1 = time.time()
        c = nn.CrossEntropyLoss()
        loss = compression_loss
        loss.backward()
        psnr = PSNR(image,data)
        acc = ACC(output,target,topk=(1,))
        print(loss.item())
        optimizer.step()

        batch_t1 = time.time()
        if i % (len(train_loader)//10) == 0:
            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; PSNR: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, i + 1,
                       len(train_loader), loss.item(),psnr.item(), bp_t1 - bp_t0, batch_t1 -
                       batch_t0))
