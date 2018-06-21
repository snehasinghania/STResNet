'''
Author: Sneha Singhania

This file contains the main program. The computation graph for ST-ResNet is built, launched in a session and trained here.
'''

from st_resnet import Graph
import tensorflow as tf
from params import Params as param
from tqdm import tqdm
from utils import batch_generator
import numpy as np
import h5py

if __name__ == '__main__': 
    # build the computation graph
    g = Graph()
    print ("Computation graph for ST-ResNet loaded\n")
    # create summary writers for logging train and test statistics
    train_writer = tf.summary.FileWriter('./logdir/train', g.loss.graph)
    val_writer = tf.summary.FileWriter('./logdir/val', g.loss.graph)   
    
    # create dummy data with correct dimensions to check if data pipeline is working
    # shape of a input map: (,ap_height, map_width, depth(num of history maps))
    x_closeness = np.random.random(size=(1000, param.map_height, param.map_width, param.closeness_sequence_length * param.nb_flow))
    x_period = np.random.random(size=(1000, param.map_height, param.map_width, param.period_sequence_length * param.nb_flow))
    x_trend = np.random.random(size=(1000, param.map_height, param.map_width, param.trend_sequence_length * param.nb_flow))
    y = np.random.random(size=(1000, param.map_height, param.map_width, 1))
    X = []   
    for j in range(x_closeness.shape[0]):
        X.append([x_closeness[j].tolist(), x_period[j].tolist(), x_trend[j].tolist()])
    
    # create train-test split of data
    train_index = int(round((0.8*len(X)),0))
    xtrain = X[:train_index]
    ytrain = y[:train_index]
    xtest = X[train_index:]
    ytest = y[train_index:]
           
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)
    
    # obtain an interator for the next batch
    train_batch_generator = batch_generator(xtrain, ytrain, param.batch_size)
    test_batch_generator = batch_generator(xtest, ytest, param.batch_size)

    print("Start learning:")
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())        
        for epoch in range(param.num_epochs):            
            loss_train = 0
            loss_val = 0
            print("Epoch: {}\t".format(epoch), )
            # training
            num_batches = xtrain.shape[0] // param.batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                x_closeness = np.array(x_batch[:, 0].tolist())
                x_period = np.array(x_batch[:, 1].tolist())
                x_trend = np.array(x_batch[:, 2].tolist())
                loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                feed_dict={g.c_inp: x_closeness,
                                                           g.p_inp: x_period,
                                                           g.t_inp: x_trend,
                                                           g.output: y_batch})               
                loss_train = loss_tr * param.delta + loss_train * (1 - param.delta)
                train_writer.add_summary(summary, b + num_batches * epoch)

            # testing
            num_batches = xtest.shape[0] // param.batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)                
                x_closeness = np.array(x_batch[:, 0].tolist())
                x_period = np.array(x_batch[:, 1].tolist())
                x_trend = np.array(x_batch[:, 2].tolist())                
                loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_inp: x_closeness,
                                                       g.p_inp: x_period,
                                                       g.t_inp: x_trend,
                                                       g.output: y_batch})
                loss_val += loss_v
                val_writer.add_summary(summary, b + num_batches * epoch)
            if(num_batches != 0):
                loss_val /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}".format(loss_train, loss_val))  
            # save the model after every epoch         
            g.saver.save(sess, param.model_path+"/current")
    train_writer.close()
    val_writer.close()
    print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
