import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-1},
        'regularization': [0.1,0.1],
        'if_optimize_regularization': False,
        'regularization_mode': 'all',
        'optimizer_name': 'SGD',
        'batch_size': 20,
        'local_size': 5,
        'device': 'cuda:0',
        'ifshow': False,
        'ifregularize': True,
        'if_GTIC': False,
        'input_datatype': torch.float,
        'target_datatype': torch.float,
        'max_num_epochs': 5,
        'min_delta': 5*1e-4,
        'patience': 5,
        'test_size': 1000,   
    }