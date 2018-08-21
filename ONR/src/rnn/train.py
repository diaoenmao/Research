import time
import os
import argparse

import numpy as np
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from data import *
from util import *

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
    '--iterations', type=int, default=1, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()
data_size = 80000
## load 32x32 patches from images
#import dataset
seed = 0
randomGen = np.random.RandomState(seed)
train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])
train_dataset,_ = fetch_dataset(data_name='ImageNet')
_,test_dataset = fetch_dataset(data_name='Kodak')
train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size,batch_size=32,num_fold=0,radomGen=randomGen)

print('total images: {}; total batches: {}'.format(
    len(train_dataset), len(train_loader)))

## load networks on GPU
import network

encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=args.lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(
        s, index))

    torch.save(binarizer.state_dict(),
               'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(
        s, index))


# resume()
output_dir = './output/img'
scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

for epoch in range(last_epoch, args.max_epochs + 1):

    scheduler.step()

    for batch, (data, _) in enumerate(train_loader):
        batch_t0 = time.time()

        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

        patches = Variable(data.cuda())

        solver.zero_grad()

        losses = []

        res = patches - 0.5

        bp_t0 = time.time()
        image = torch.zeros(data.size(0), 3, 32, 32) + 0.5
        for _ in range(args.iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            codes = binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            image = image + output.data.cpu()
            res = res - output
            losses.append(res.abs().mean())

        bp_t1 = time.time()

        loss = sum(losses) / args.iterations
        loss.backward()
        psnr = PSNR(image,data)
        solver.step()

        batch_t1 = time.time()
        if batch % (len(train_loader)//5) == 0:
            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; PSNR: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, batch + 1,
                       len(train_loader), loss.item(),psnr.item(), bp_t1 - bp_t0, batch_t1 -
                       batch_t0))
        # print(('{:.4f} ' * args.iterations +
               # '\n').format(* [l.item() for l in losses]))

        #index = (epoch - 1) * len(train_loader) + batch

        # ## save checkpoint every 500 training steps
        # if index % 500 == 0:
            # save(0, False)
    losses = Meter()
    psnrs = Meter()
    for batch, (data, _) in enumerate(test_loader):
        batch_t0 = time.time()

        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

        patches = Variable(data.cuda())

        cur_losses = []

        res = patches - 0.5

        bp_t0 = time.time()
        image = torch.zeros(data.size(0), 3, 32, 32) + 0.5
        for _ in range(args.iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            codes = binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            image = image + output.data.cpu()
            res = res - output
            cur_losses.append(res.abs().mean())

        bp_t1 = time.time()
        tmp = nn.MSELoss()
        loss = sum(cur_losses) / args.iterations
        psnr = PSNR(image,data)
        losses.update(loss.item(), data.size(0))
        psnrs.update(psnr.item(), data.size(0))
        batch_t1 = time.time()
        if epoch % 3 == 0:
            if not os.path.exists('.{}/image_{}.png'.format(output_dir,batch)):
                save_img(data,'{}/image_{}.png'.format(output_dir,batch))
            save_img(image,'{}/image_{}_{}.png'.format(output_dir,batch,epoch))

    print(
        '[TEST] Epoch[{}]; Loss: {:.6f}; PSNR: {:.6f}'.
        format(epoch, losses.avg, psnrs.avg))
    #save(epoch)
