import numpy as np
import torchvision.transforms as transforms
from util import *

class Codec():
    def __init__(model):
        self.model = model
        self.model.eval()
        self._transform = transforms.Compose([
                transforms.Lambda(lambda x: RGB_to_YCbCr(x))
                transforms.ToTensor()
            ])     
        self._detransform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: YCbCr_to_RGB(x))
        ])
        
    def transform(self, x):
        transformed_x = self._transform(x)
        return transformed_x
        
    def detransform(self, decoded_x):
        detransformed_x = self._detransform(decoded_x)
        return detransformed_x
    
    def entropy_code(self,code):
        return
        
    def entropy_decode(self,entropy_code):
        return
        
    def encode(self, x):
        code = self.model.code(x)
        return code
        
    def decode(self, code):
        decoded_x = self.model.decode(code)
        return decoded_x    