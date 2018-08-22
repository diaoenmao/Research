import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        'optimizer_name': 'Adam',
        'milestones': [10, 20, 50, 100],
        'gamma': 0.5,
        'patch_shape': (32,32),
        'batch_size': 32,
        'data_size': 1000,
        'device': 'cuda',
        'max_num_epochs': 100,
        'save_mode': 1,
    }