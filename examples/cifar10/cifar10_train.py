# -*- coding: utf-8 -*-
"""
This is an example of training cifar10.

**NOTE**: To run this script, you should install keras, as the data handling utils is from keras.
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle
import random
import argparse
import subprocess
import numpy as np
from datetime import datetime

import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import nics_fix as nf

FLAGS = None

class RandomCrop():
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (np.array): Image to be cropped.
        Returns:
            np.array: Cropped image.
        """
        img_padding = img

        if self.padding > 0:
            img_padding = np.zeros((img.shape[0], self.size + self.padding * 2, self.size + self.padding * 2, img.shape[-1]))
            img_padding[:, self.padding:self.size+self.padding, self.padding:self.size+self.padding, :] = img

        w, h = img_padding.shape[1:3]
        th = tw = self.size
        if w == tw and h == th:
            return img_padding

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img_padding[:, x1:x1 + tw, y1:y1 + th, :]

def Conv4Dense2(x, num_classes, training, *args, **kwargs):
    return (nf.wrap(x)
            .Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv1")
            .ReLU(name="relu1")
            .Conv2D(filters=32, kernel_size=(3, 3), name="conv2")
            .ReLU(name="relu2")
            .MaxPool(pool_size=(2, 2), strides=2, name="pool2")
            .Dropout(rate=0.25, training=training)
            .Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv3")
            .ReLU(name="relu3")
            .Conv2D(filters=64, kernel_size=(3, 3), name="conv4")
            .ReLU(name="relu4")
            .MaxPool(pool_size=(2, 2), strides=2, name="pool4")
            .Dropout(rate=0.25, training=training)
            .Flatten()
            .Dense(units=512, name="dense5")
            .ReLU(name="relu5")
            .Dropout(rate=0.5, training=training)
            .Dense(units=num_classes, name="dense6"))

def VGG11(x, num_classes, training, weight_decay):
    with nf.kwargs_scope_by_type(Conv2D={"padding": "same",
                                         "kernel_size": (3, 3),
                                         "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                         "kernel_initializer": tf.contrib.layers.variance_scaling_initializer()},
                                 MaxPool={"pool_size": (2, 2),
                                          "strides": 2},
                                 Dense={
                                     "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                     "kernel_initializer": tf.contrib.layers.xavier_initializer()}):
        return (nf.wrap(x)
                .Conv2D(filters=64, name="conv1")
                .ReLU(name="relu1")
                .MaxPool(name="pool1")
                .Conv2D(filters=128, name="conv2")
                .ReLU(name="relu2")
                .MaxPool(name="pool2")
                .Conv2D(filters=256, name="conv3_1")
                .ReLU(name="relu3_1")
                .Conv2D(filters=256, name="conv3_2")
                .ReLU(name="relu3_2")
                .MaxPool(name="pool3")
                .Conv2D(filters=512, name="conv4_1")
                .ReLU(name="relu4_1")
                .Conv2D(filters=512, name="conv4_2")
                .ReLU(name="relu4_2")
                .MaxPool(name="pool4")
                .Conv2D(filters=512, name="conv5_1")
                .ReLU(name="relu5_1")
                .Conv2D(filters=512, name="conv5_2")
                .ReLU(name="relu5_2")
                .MaxPool(name="pool5")
                .Flatten()
                .Dense(units=512, name="dense6")
                .ReLU(name="relu6")
                .Dropout(rate=0.5, training=training)
                .Dense(units=512, name="dense7")
                .ReLU(name="relu7")
                .Dropout(rate=0.5, training=training)
                .Dense(units=num_classes, name="dense8"))

def main(_):
    batch_size = FLAGS.batch_size # default to 128
    num_classes = 10

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], " train samples")
    print(x_test.shape[0], " test samples")
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    
    # Construct the model
    cfgs = nf.parse_cfg_from_str("")
    if FLAGS.cfg is not None:
        cfgs = nf.parse_cfg_from_file(FLAGS.cfg)
    
    x = tf.placeholder(tf.float32, shape=[None] + list(x_train.shape[1:]))
    labels = tf.placeholder(tf.float32, [None, num_classes])
    weight_decay = FLAGS.weight_decay
    
    with nf.fixed_scope("fixed_mlp_mnist", cfgs) as (s, training, fixed_mapping):
        training_placeholder = training
        # Using chaining writing style:
        logits = globals()[FLAGS.model](x, num_classes, training, weight_decay).tensor
    
    # Construct the fixed saver
    saver = nf.utils.fixed_model_saver(fixed_mapping)
    
    # Loss and metrics
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    index_label = tf.argmax(labels, 1)
    correct = tf.equal(tf.argmax(logits, 1), index_label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    top5_correct = tf.nn.in_top_k(logits, index_label, 5)
    top5_accuracy = tf.reduce_mean(tf.cast(top5_correct, tf.float32))
    
    # Initialize RMSprop optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-6)
    train_step = optimizer.minimize(cross_entropy)
    
    print("Using real-time data augmentation.")
    # This will do preprocessing and realtime data augmentation:
    datagen_train = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen_test = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen_train.fit(x_train)
    datagen_test.fit(x_test)
    random_crop = RandomCrop(32, 4) # padding 4 and crop 32x32
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start training...")
        # Training
        for epoch in range(1, FLAGS.epochs+1):
            start_time = time.time()
            gen = datagen_train.flow(x_train, y_train, batch_size=batch_size)
            steps_per_epoch = x_train.shape[0] // batch_size
            loss_v_epoch = 0
            acc_1_epoch = 0
            acc_5_epoch = 0

            # Train batches
            for step in range(1, steps_per_epoch+1):
                # TODO: use another thread to execute the data augumentation and enqueue
                x_v, y_v = next(gen)
                x_crop_v = random_crop(x_v)
                _, loss_v, acc_1, acc_5 = sess.run([train_step, cross_entropy, accuracy, top5_accuracy],
                                                   feed_dict={
                                                       x: x_crop_v,
                                                       labels: y_v
                                                   })
                print("\rEpoch {}: steps {}/{}".format(epoch, step, steps_per_epoch), end="")
                loss_v_epoch += loss_v
                acc_1_epoch += acc_1
                acc_5_epoch += acc_5
                
            loss_v_epoch /= steps_per_epoch
            acc_1_epoch /= steps_per_epoch
            acc_5_epoch /= steps_per_epoch

            duration = time.time() - start_time
            sec_per_batch = duration / (steps_per_epoch * batch_size)
            print("\r{}: Epoch {}; (average) loss: {:.3f}; (average) top1 accuracy: {:.2f} %; (average) top5 accuracy: {:.2f} %. {:.3f} sec/batch"\
                  .format(datetime.now(), epoch, loss_v_epoch, acc_1_epoch * 100, acc_5_epoch * 100, sec_per_batch))
            # End training batches

            # Test on the validation set
            if epoch % FLAGS.test_frequency == 0:
                test_gen = datagen_test.flow(x_test, y_test, batch_size=batch_size)
                steps_per_epoch = x_test.shape[0] // batch_size
                loss_test = 0
                acc_1_test = 0
                acc_5_test = 0
                for step in range(1, steps_per_epoch+1):
                    x_v, y_v = next(test_gen)
                    loss_v, acc_1, acc_5 = sess.run([cross_entropy, accuracy, top5_accuracy],
                                            feed_dict={
                                                x: x_v,
                                                labels: y_v
                                            })
                    print("\r\ttest steps: {}/{}".format(step, steps_per_epoch), end="")
                    loss_test += loss_v
                    acc_1_test += acc_1
                    acc_5_test += acc_5
                loss_test /= steps_per_epoch
                acc_1_test /= steps_per_epoch
                acc_5_test /= steps_per_epoch
                print("\r\tTest: loss: {}; top1 accuracy: {:.2f} %; top5 accuracy: {:2f} %.".format(loss_test, acc_1_test * 100, acc_5_test * 100))
            # End test on the validation set
        # End training

        if FLAGS.train_dir:
            if not os.path.exists(FLAGS.train_dir):
                subprocess.check_call("mkdir -p {}".format(FLAGS.train_dir),
                                      shell=True)
            print("Saved model to: ", saver.save(sess, FLAGS.train_dir))
    
    # # Load label names to use in prediction results
    # label_list_path = "datasets/cifar-10-batches-py/batches.meta"
    # keras_dir = os.path.expanduser(os.path.join("~", ".keras"))
    # datadir_base = os.path.expanduser(keras_dir)
    # if not os.access(datadir_base, os.W_OK):
    #     datadir_base = os.path.join("/tmp", ".keras")
    # label_list_path = os.path.join(datadir_base, label_list_path)
    
    # with open(label_list_path, mode="rb") as f:
    #     labels = pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, default="",
                        help="Directory for storing snapshots")
    parser.add_argument("--model", type=str, default="VGG11",
                        choices=["VGG11", "Conv4Dense2"], help="The network structure.")
    parser.add_argument("--cfg", type=str, default=None,
                        help="The file for fixed configration.")
    parser.add_argument("--test_frequency", type=int, default=5,
                        metavar="N", help="Test the accuracies on validation set "
                        "after every N epochs.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="The max training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The training/testing batch size.")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="The L2 weight decay parameter.")
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.train_dir:
        print("WARNING: model will not be saved if `--train_dir` option is not given.")
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

