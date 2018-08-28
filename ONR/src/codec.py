import torch
import numpy as np


class Codec():
    def __init__(self,model,device):
        self.model = model
        self.model.eval()
        self.device = device
        
    def entropy_encode(self,code):
        code = (np.stack(code).astype(np.int8) + 1) // 2
        code = np.packbits(code.reshape(-1))
        return code
        
    def entropy_decode(self,code,batch_size):
        code = np.unpackbits(code)
        code = np.reshape(code, (-1,batch_size,32,2,2)).astype(np.float32) * 2 - 1
        code = torch.from_numpy(code).to(device)
        return code
        
    def encode(self, x):
        code = self.model.encode(x)
        code = self.entropy_encode(code)
        return code
        
    def decode(self,code,batch_size):
        code = self.entropy_decode(code,batch_size)
        decoded_x = self.model.decoder(code,batch_size)
        return decoded_x    