import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'Adam',
        'batch_size': 1,
        'data_size': 10000,
        'device': 'cpu',
        'max_num_epochs': 1,
        'save_mode': 1,
    }