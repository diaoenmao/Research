import torch

def init():
    global PARAM
    PARAM = {
        'model_data_name': 'EMNIST',
        'train_data_name': 'EMNIST',
        'test_data_name': 'EMNIST',
        'model_dir': 'emnist',
        'model_name': 'Joint_global',
        'resume_TAG': '',
        'special_TAG': '',
        'lr': 1e-1,
        'milestones': [150,250],
        'gamma': 0.1,
        'branch': False,
        'balance': False,
        'batch_size': 128,
        'num_workers': 4,
        'data_size': 0,
        'device': 'cuda:0',
        'device_ids': [0,1],
        'max_num_epochs': 350,
        'save_mode': 0,
        'world_size': 1,
        'meter_names': ['loss','acc','psnr','batch_time'],
        'topk': 1,
        'num_Experiments': 1,
        'tuning_param': {'compression': 0, 'classification': 1},
        'num_iter': 16,
        'max_depth': 3,
        'max_channel': 256,
        'patch_shape': (32,32),
        'step': [1.0,1.0],
        'jump_rate': 2,
        'num_nodes': 2,
        'resume_mode': 0
    }