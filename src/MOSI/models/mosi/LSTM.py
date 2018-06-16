import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def passthrough(x, **kwargs):
    return x

class _LSTM(nn.Module):
    def __init__(self,input_feature,output_feature=10):
        super(_LSTM, self).__init__()
        self.hidden_dim=128
        self.bn1 = nn.BatchNorm1d(input_feature)
        self.rnn1 = nn.LSTM(input_feature,self.hidden_dim, bidirectional=False)
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(self.hidden_dim, output_feature)
        
    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device='cuda:0'),
                torch.zeros(1, batch_size, self.hidden_dim, device='cuda:0'))
                
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous() # N,C,L
        x = self.bn1(x)
        x = x.permute(2, 0, 1).contiguous() # L,N,C
        lstm_out, self.hidden = self.rnn1(x, self.hidden) 
        x = self.dropout1(lstm_out[-1])
        x = self.fc1(x)
        return x
        
        
def lstm(**kwargs):
    model = _LSTM(**kwargs)
    return model  