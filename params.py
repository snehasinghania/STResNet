'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file contains class Params for hyperparameter declarations
'''

class Params(object):
    batch_size = 32
    map_height = 75
    map_width = 73
    closeness_sequence_length = 12
    period_sequence_length = 24
    trend_sequence_length =  8
    num_of_filters = 64
    num_of_residual_units = 3
    num_of_output = 1
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.001
    num_epochs = 10
    logdir = "train"
