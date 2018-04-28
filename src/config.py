import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-1},
        'regularization': [-7,-7],
        'if_optimize_regularization': True,
        'regularization_mode': 'single',
        'optimizer_name': 'SGD',
        'batch_size': 20,
        'local_size': 4,
        'device': 'cuda:0',
        'verbose': True,
        'ifshow': True,
        'ifregularize': True,
        'if_GTIC': True,
        'input_datatype': torch.float,
        'target_datatype': torch.long,
        'max_num_epochs': 10,
        'min_delta': 5*1e-4,
        'patience': 5,
        'test_size': 0.1,
        'output_feature': 2      
    }