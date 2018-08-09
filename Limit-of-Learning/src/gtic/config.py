import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'LBFGS',
        'batch_size': 30,
        'data_size': 10000,
        'device': 'cuda',
        'max_num_epochs': 3,
        'save_mode': 1,
    }