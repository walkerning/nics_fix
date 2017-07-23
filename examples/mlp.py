# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nics_fix as nf

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    training_placeholder = None
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    with nf.fixed_scope("fixed_mlp_mnist", nf.parse_cfg_from_str("")) as (s, training):
        training_placeholder = training
        #res = nf.wrap(x).Dense(units=100).ReLU().Dense(units=10)
        res = nf.wrap(x).Dense(units=10)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=res.tensor))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    train_step = optimizer.apply_gradients(grads_and_vars)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, training_placeholder: True})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(res.tensor, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels,
                                        training_placeholder: False}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

