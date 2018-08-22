import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
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
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)

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
        self.conv = Conv()
        self.decoder = DecoderCell()
        self.compression_loss_fn = nn.L1Loss()
        self.init_hidden()
        
    def init_hidden(self, batch_size):
        encoder_h_1 = (torch.zeros(batch_size, 256, 8, 8),
                       torch.zeros(batch_size, 256, 8, 8))
        encoder_h_2 = (torch.zeros(batch_size, 512, 4, 4),
                       torch.zeros(batch_size, 512, 4, 4))
        encoder_h_3 = (torch.zeros(batch_size, 512, 2, 2),
                       torch.zeros(batch_size, 512, 2, 2))

        decoder_h_1 = (torch.zeros(batch_size, 512, 2, 2),
                       torch.zeros(batch_size, 512, 2, 2))
        decoder_h_2 = (torch.zeros(batch_size, 512, 4, 4),
                       torch.zeros(batch_size, 512, 4, 4))
        decoder_h_3 = (torch.zeros(batch_size, 256, 8, 8),
                       torch.zeros(batch_size, 256, 8, 8))
        decoder_h_4 = (torch.zeros(batch_size, 128, 16, 16),
                       torch.zeros(batch_size, 128, 16, 16))
        self.encoder_h = [encoder_h_1,encoder_h_2,encoder_h_3]
        self.decoder_h = [decoder_h_1,decoder_h_2,decoder_h_3,decoder_h_4]
        return self.encoder_h, self.decoder_h
                       
    def forward(self, x, hidden=None):
        if(hidden is None):
            self.init_hidden(x.size(0))
        else:
            self.encoder_h,self.deoder_h = hidden[0],hidden[1]
        output_0 = x.new_zeros(x.size())
        res = []
        codes = []
        for _ in range(self.num_iter):
            encoded_x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2] = self.encoder(
                    x, self.encoder_h[0], self.encoder_h[1], self.encoder_h[2])
            code = self.binarizer(encoded_x)
            decoded_x, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
                code, self.decoder_h[0], self.decoder_h[1], self.decoder_h[2], self.decoder_h[3])
            res.append(self.compression_loss_fn(decoded_x,x))
            codes.append(code)
            output_0 = output_0 + decoded_x
            x = x - decoded_x
        codes = torch.cat(codes,1)
        output_1 = self.conv(codes)
        output_2 = sum(res)/self.num_iter
        return [output_0,output_1,output_2]
    
    
def Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(
            512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,10)
        
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





        
    
    