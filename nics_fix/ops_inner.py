# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from nics_fix.ops import fixed_register, no_fixed_register

Conv2D = fixed_register(tf.layers.conv2d, "Conv2D")
Dense = fixed_register(tf.layers.dense, "Dense")
ReLU = fixed_register(tf.nn.relu, "ReLU")
Split = fixed_register(tf.split, "Split")

MaxPool = no_fixed_register(tf.layers.max_pooling2d, "MaxPool")
Dropout = no_fixed_register(tf.layers.dropout, "Dropout")
Flatten = no_fixed_register(tf.contrib.layers.flatten, "Flatten")
