import torch

def init():
    global PARAM
    PARAM = {
        'optimizer_param': {'lr': 0.8},
        'regularization_parameters': [1,0.001,0.001],
        'if_joint_regularization': True,
        'optimizer_name': 'LBFGS',
        'batch_size': 20,
        'ifcuda': True,
        'verbose': True,
        'ifsave': True,
        'ifshow': False,
        'ifregularize': True,
        'input_datatype': torch.FloatTensor,
        'target_datatype': torch.LongTensor,
        'max_num_epochs': 5,
        'min_delta': 5*1e-4,
        'patience': 5,
        'test_size': 0.1,
        'output_feature': 2      
    }