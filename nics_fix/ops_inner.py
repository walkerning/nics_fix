# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from nics_fix.ops import fixed_register, no_fixed_register

Conv2D = fixed_register(tf.layers.conv2d, "Conv2D")
Dense = fixed_register(tf.layers.dense, "Dense")
ReLU = fixed_register(tf.nn.relu, "ReLU")
Tanh = fixed_register(tf.nn.tanh, "Tanh")
Split = fixed_register(tf.split, "Split")
#BN = no_fixed_register(tf.layers.batch_normalization, "BN")
BN = no_fixed_register(tf.contrib.layers.batch_norm, "BN")

MaxPool = no_fixed_register(tf.layers.max_pooling2d, "MaxPool")
AvePool = no_fixed_register(tf.layers.average_pooling2d, "AvePool")
Dropout = no_fixed_register(tf.layers.dropout, "Dropout")
Flatten = no_fixed_register(tf.contrib.layers.flatten, "Flatten")

BN_DECAY = 0.9997
BN_EPSILON = 0.001
"""
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

def BN(x, is_training, use_bias, name):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias == True:
        bias = _get_variable(name+'bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable(name+'beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable(name+"gamma",
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable(name+'moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable(name+'moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x
"""