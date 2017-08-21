# -*- coding: utf-8 -*-
"""
This is an example of loading trained weights and running evaluation only.
"""

from __future__ import print_function

import sys
import argparse
import itertools

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nics_fix as nf

FLAGS = None

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    training_placeholder = None
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cfgs = nf.parse_cfg_from_str("")
    if FLAGS.cfg is not None:
        cfgs = nf.parse_cfg_from_file(FLAGS.cfg)

    with nf.fixed_scope("fixed_mlp_mnist", cfgs) as (s, training, fixed_mapping):
        training_placeholder = training
        # Using chaining writing style:
        res = nf.wrap(x).Dense(units=100, name="dense1").ReLU(name="relu1").Dense(units=10, name="dense2").tensor
        # Alternatively, you can use the normal writing style:
        # res = nf.Dense(nf.ReLU(nf.Dense(x, units=100, name="dense1"), name="relu1"), units=10, name="dense2")

    saver = nf.utils.fixed_model_saver(fixed_mapping)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=res))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    train_step = optimizer.apply_gradients(grads_and_vars)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Restore the trained weights from snapshot
    saver.restore(sess, FLAGS.train_dir)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    weight_data_scales = tf.get_collection(nf.FixedKeys.FIXED_WEIGHT_DATA_SCALE)
    act_data_scales = tf.get_collection(nf.FixedKeys.FIXED_ACTIVATION_DATA_SCALE)
    acc, weight_scales, act_scales = sess.run([accuracy, weight_data_scales, act_data_scales],
                                              feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,
                                                         training_placeholder: False})

    print("accuracy: ", acc)
    print("Data fix scales: ", "\n".join(["{} {}".format(*item) for item in \
                                          zip([tensor.op.name for tensor in \
                                               itertools.chain(weight_data_scales, act_data_scales)],
                                              itertools.chain(weight_scales, act_scales))]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str,
                        help="Directory for restoring snapshots")
    parser.add_argument("--data_dir", type=str, default="/tmp/tensorflow/mnist/input_data",
                        help="Directory for storing input data")
    parser.add_argument("--cfg", type=str, default=None,
                        help="The file for fixed configration.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

