import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'Adam',
        'batch_size': 30,
        'data_size': 1000,
        'device': 'cuda',
        'max_num_epochs': 50,
        'save_mode': 1,
    }