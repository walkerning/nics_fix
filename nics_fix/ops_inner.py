# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from nics_fix.ops import fixed_register

Conv2d = fixed_register(tf.layers.conv2d, "Conv2d")
Dense = fixed_register(tf.layers.dense, "Dense")
ReLU = fixed_register(tf.nn.relu, "ReLU")
Split = fixed_register(tf.split, "Split")
