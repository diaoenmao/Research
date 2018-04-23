import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-1},
        'regularization_parameters': [0,-7,-7],
        'if_joint_regularization': False,
        'optimizer_name': 'SGD',
        'batch_size': 20,
        'local_size': 2,
        'ifcuda': True,
        'verbose': True,
        'ifsave': True,
        'ifshow': True,
        'ifregularize': True,
        'input_datatype': torch.FloatTensor,
        'target_datatype': torch.LongTensor,
        'max_num_epochs': 10,
        'min_delta': 5*1e-4,
        'patience': 5,
        'test_size': 0.1,
        'output_feature': 2      
    }