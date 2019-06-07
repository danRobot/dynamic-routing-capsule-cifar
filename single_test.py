'''
Created on 5/06/2019

@author: mahou
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.advanced_activations import LeakyReLU


config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction =0.9
set_session(tf.Session(config=config))
import sys
from sklearn.metrics import confusion_matrix

from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Flatten,
    Reshape,
    Dropout
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras.models import Model
from models.capsule_layers import CapsuleLayer, PrimaryCapsule, Length,Mask
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import optimizers
from utils.helper_function import load_cifar_10,load_cifar_100
from models.capsulenet import CapsNet as CapsNetv1
import numpy as np
import time as t
import os
from sys import exit


def test_model(model_path,x_test,y_test):
    num_classes=11
    with tf.Session() as sess:
        y_test=sess.run(tf.one_hot(y_test,num_classes))
    print(y_test.shape)
    model = CapsNetv1(input_shape=[200,200, 3],
                      n_class=num_classes,
                      n_route=3,
                      kth=True)
    model.load_weights(filepath=model_path)
    y_pred, _ = model.predict([x_test, y_test], batch_size=128)
    ac=np.argmax(y_pred, 1) == np.argmax(y_test, 1)
    print(y_pred)
    print(y_test)
    print(np.argmax(y_pred, 1))
    print(np.argmax(y_test, 1))
    if(ac[0]):
        print('Correcto')
    else:
        print('Incorrecto')
    
    pass