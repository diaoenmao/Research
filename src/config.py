import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'SGD',
        'batch_size': 100,
        'data_size': 50000,
        'device': 'cuda:0',
        'if_resume': False,
        'if_show': True,
        'verbose': False,
        'max_num_epochs': 300,
    }