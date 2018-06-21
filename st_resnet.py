'''
Author: Sneha Singhania

This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture written in an OOP and modular manner. The outline of the architecture from inputs to outputs in defined here using calls to functions defined in modules.py to handle the inner complexity. Modularity ensures that the working of a component can be easily modified in modules.py without changing the skeleton of the ST-ResNet architecture defined in this file.
'''

from params import Params as param
import modules as my
import tensorflow as tf
import numpy as np

class Graph(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            B, H, W, C, P, T, O, F, U  = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length*param.nb_flow, param.period_sequence_length*param.nb_flow, param.trend_sequence_length*param.nb_flow, param.num_of_output ,param.num_of_filters, param.num_of_residual_units,            
            # get input and output          
            # shape of a input map: (Batch_size, map_height, map_width, depth(num of history maps))
            self.c_inp = tf.placeholder(tf.float32, shape=[B, H, W, C], name="closeness")
            self.p_inp = tf.placeholder(tf.float32, shape=[B, H, W, P], name="period")
            self.t_inp = tf.placeholder(tf.float32, shape=[B, H, W, T], name="trend")
            self.output = tf.placeholder(tf.float32, shape=[B, H, W, O], name="output") 
            
            # ResNet architecture for the three modules
            # module 1: capturing closeness (recent)
            self.closeness_output = my.ResInput(inputs=self.c_inp, filters=F, kernel_size=(7, 7), scope="closeness_input", reuse=None)
            self.closeness_output = my.ResNet(inputs=self.closeness_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=None)
            self.closeness_output = my.ResOutput(inputs=self.closeness_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=None)            
            # module 2: capturing period (near)
            self.period_output = my.ResInput(inputs=self.p_inp, filters=F, kernel_size=(7, 7), scope="period_input", reuse=None)
            self.period_output = my.ResNet(inputs=self.period_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=True)
            self.period_output = my.ResOutput(inputs=self.period_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=True)            
            # module 3: capturing trend (distant) 
            self.trend_output = my.ResInput(inputs=self.t_inp, filters=F, kernel_size=(7, 7), scope="trend_input", reuse=None)
            self.trend_output = my.ResNet(inputs=self.trend_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=True)
            self.trend_output = my.ResOutput(inputs=self.trend_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=True)            
            # parameter matrix based fusion
            self.x_res = my.Fusion(self.closeness_output, self.period_output, self.trend_output, scope="fusion", shape=[W, W])                        
            # loss function
            self.loss = tf.reduce_sum(tf.pow(self.x_res - self.output, 2)) / tf.cast((self.x_res.shape[0]), tf.float32)            
            # use Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon).minimize(self.loss)           
            #loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()            
            self.saver = tf.train.Saver(max_to_keep=None)
