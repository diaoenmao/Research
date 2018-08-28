import torch

def init():
    global PARAM
    PARAM = {
        'milestones': [10, 20, 50, 100],
        'gamma': 0.5,
        'patch_shape': (32,32),
        'batch_size': 30,
        'data_size': 60000,
        'device': 'cuda',
        'max_num_epochs': 200,
        'save_mode': 2,
        'if_resume': False
    }