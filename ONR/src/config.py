import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'Adam',
        'milestones': [10],
        'patch_shape': (128,128),
        'batch_size': 32,
        'data_size': 1000,
        'device': 'cuda',
        'max_num_epochs': 30,
        'save_mode': 1,
    }