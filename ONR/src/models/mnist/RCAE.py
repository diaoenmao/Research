import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from modules import ConvLSTMCell, Sign
config.init()
device = torch.device(config.PARAM['device'])

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, x, hidden1, hidden2, hidden3):
        x = self.conv(x)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, x):
        feat = self.conv(x)
        x = torch.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            32, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(x)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4

class RCAE(nn.Module):
    def __init__(self,num_iter=16):
        super(RCAE, self).__init__()
        self.num_iter = num_iter
        self.encoder = EncoderCell()
        self.binarizer = Binarizer()
        self.net = Net()
        self.decoder = DecoderCell()
        
    def compression_loss_fn(self,output,target):
        res = (output-target).abs().mean()
        return res

    def encode(self, x, hidden=None):
        if(hidden is None):
            self.encoder_h,self.decoder_h = self.init_hidden(x.size(0))
        else:
            self.encoder_h,self.decoder_h = hidden[0],hidden[1]
        codes = []
        image = x.new_zeros(x.size())
        for i in range(self.num_iter):
            encoded_x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2] = self.encoder(
                    x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2])
            code = self.binarizer(encoded_x)
            decoded_x, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3] = self.decoder(
                code, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3])
            image = image + decoded_x
            codes.append(code)
            x = x - decoded_x
        return codes
    
    def decode(self, codes, batch_size, hidden=None):
        if(hidden is None):
            self.encoder_h,self.decoder_h = self.init_hidden(batch_size)
        else:
            self.encoder_h,self.decoder_h = hidden[0],hidden[1]
 
        image = torch.zeros(batch_size, 1, 32, 32, device=device)
        for i in range(min(self.num_iter, len(codes))):
            decoded_x, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3] = self.decoder(
                codes[i], self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3])
            image = image + decoded_x
        return image
        
    def init_hidden(self, batch_size):
        encoder_h_1 = (torch.zeros(batch_size, 256, 8, 8, device = device),
                       torch.zeros(batch_size, 256, 8, 8, device = device))
        encoder_h_2 = (torch.zeros(batch_size, 512, 4, 4, device = device),
                       torch.zeros(batch_size, 512, 4, 4, device = device))
        encoder_h_3 = (torch.zeros(batch_size, 512, 2, 2, device = device),
                       torch.zeros(batch_size, 512, 2, 2, device = device))

        decoder_h_1 = (torch.zeros(batch_size, 512, 2, 2, device = device),
                       torch.zeros(batch_size, 512, 2, 2, device = device))
        decoder_h_2 = (torch.zeros(batch_size, 512, 4, 4, device = device),
                       torch.zeros(batch_size, 512, 4, 4, device = device))
        decoder_h_3 = (torch.zeros(batch_size, 256, 8, 8, device = device),
                       torch.zeros(batch_size, 256, 8, 8, device = device))
        decoder_h_4 = (torch.zeros(batch_size, 128, 16, 16, device = device),
                       torch.zeros(batch_size, 128, 16, 16, device = device))
        encoder_h = [encoder_h_1,encoder_h_2,encoder_h_3]
        decoder_h = [decoder_h_1,decoder_h_2,decoder_h_3,decoder_h_4]
        return encoder_h, decoder_h
                       
    def forward(self, x, hidden=None):
        if(hidden is None):
            self.encoder_h,self.decoder_h = self.init_hidden(x.size(0))
        else:
            self.encoder_h,self.decoder_h = hidden[0],hidden[1]
        res = []
        codes = []
        image = x.new_zeros(x.size())
        for i in range(self.num_iter):
            encoded_x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2] = self.encoder(
                    x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2])
            code = self.binarizer(encoded_x)
            decoded_x, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3] = self.decoder(
                code, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3])
            image = image + decoded_x
            res.append(self.compression_loss_fn(decoded_x,x))
            codes.append(code)
            x = x - decoded_x
        compression_loss = sum(res)/self.num_iter
        code = torch.cat(codes,1)
        output = self.net(code)
        return compression_loss,image,output
    
    
class Net(nn.Module):
    def __init__(self,num_iter=16):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            num_iter*32, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,10)
        
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





        
    
    