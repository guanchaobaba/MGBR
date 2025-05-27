#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONFIG = {
    'name': '@gc',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'MGBR',
    'dataset_name': 'Youshu',
    'task': 'test',

    ## search hyperparameters
    'lrs': [1e-3],
    'message_dropouts': [0, 0.1],
    'node_dropouts': [0, 0.1],
    'decays': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],

    ## optimal hyperparameters 
    # 'lrs': [3e-4],
    # 'message_dropouts': [0.3],
    # 'node_dropouts': [0],
    # 'decays': [1e-7],

    'sample': 'simple',
     #'sample': 'hard',


    ## other settings
    'epochs': 300,
    'early': 150,
    'log_interval': 8,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test':['./model_file_from_simple_sample.pth']
}

