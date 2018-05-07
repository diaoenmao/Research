import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-1},
        'reg_optimizer_param': {'lr': 1e-3},
        'regularization': None,
        'if_optimize_regularization': False,
        'regularization_mode': 'all',
        'optimizer_name': 'SGD',
        'batch_size': 20,
        'local_size': 100,
        'device': 'cuda:0',
        'if_show': False,
        'if_regularize': True,
        'if_GTIC': True,
        'if_load': False,
        'if_save': False,
        'if_verbose': False,
        'input_datatype': torch.float,
        'target_datatype': torch.float,
        'max_num_epochs': 5,
        'min_delta': 5*1e-4,
        'patience': 5,
        'test_size': 1000,   
    }