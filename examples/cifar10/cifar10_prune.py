# -*- coding: utf-8 -*-
"""
This is an example of training cifar10.

Hyper-parameters for training VGG11 follows https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py

**NOTE**: To run this script, you should install keras, as the data handling utils is from keras.
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import yaml
import random
import argparse
import subprocess
import multiprocessing
import numpy as np
from datetime import datetime

import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import nics_fix as nf

FLAGS = None

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

prune_rate = {"fixed_mlp_mnist/conv1/conv1/kernel:0": 0.1, \
              "fixed_mlp_mnist/conv2/conv2/kernel:0": 0.4, \
              "fixed_mlp_mnist/conv3_1/conv3_1/kernel:0": 0.3, \
              "fixed_mlp_mnist/conv3_2/conv3_2/kernel:0": 0.4, \
              "fixed_mlp_mnist/conv4_1/conv4_1/kernel:0": 0.5, \
              "fixed_mlp_mnist/conv4_2/conv4_2/kernel:0": 0.7, \
              "fixed_mlp_mnist/conv5_1/conv5_1/kernel:0": 0.9, \
              "fixed_mlp_mnist/conv5_2/conv5_2/kernel:0": 0.9, \
              "fixed_mlp_mnist/dense6/dense6/kernel:0": 0.9, \
              "fixed_mlp_mnist/dense7/dense7/kernel:0": 0.9, \
              "fixed_mlp_mnist/dense8/dense8/kernel:0": 0.9}

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

class MultiProcessGen(object):
    def __init__(self, data_gen, max_queue_size=10, wait_time=0.05):
        self.data_gen = data_gen
        self.wait_time = wait_time
        self.queue = multiprocessing.Queue(maxsize=max_queue_size)
        self._stop_event = multiprocessing.Event()
        def datagen_task():
            while not self._stop_event.is_set():
                try:
                    if self.queue.qsize() < max_queue_size:
                        self.queue.put(next(self.data_gen))
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise
        self.thread = multiprocessing.Process(target=datagen_task)
        self.thread.start()
        self.running = True

    def stop(self):
        if self.running:
            self._stop_event.set()
            self.thread.terminate()
            self.queue.close()
            self.thread = None
            self._stop_event = None
            self.queue = None
            self.running = False

    def __iter__(self):
        return self

    def next(self):
        while self._stop_event and not self._stop_event.is_set():
            if not self.queue.empty():
                return self.queue.get()
            else:
                time.sleep(self.wait_time)
        # return self.queue.get(block=True)

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

def log(*args, **kwargs):
    flush = kwargs.pop("flush", None)
    if FLAGS.log_file is not None:
        print(*args, file=FLAGS.log_file, **kwargs)
        if flush:
            FLAGS.log_file.flush()
    print(*args, **kwargs)
    if flush:
        sys.stdout.flush()

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    batch_size = FLAGS.batch_size # default to 128
    num_classes = 10

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    log(x_train.shape[0], " train samples")
    log(x_test.shape[0], " test samples")
    
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
    
    s_cfgs = nf.parse_strategy_cfg_from_str("")
    if FLAGS.scfg is not None:
        s_cfgs = nf.parse_strategy_cfg_from_file(FLAGS.scfg)

    x = tf.placeholder(tf.float32, shape=[None] + list(x_train.shape[1:]))
    labels = tf.placeholder(tf.float32, [None, num_classes])
    weight_decay = FLAGS.weight_decay
    
    with nf.fixed_scope("fixed_mlp_mnist", cfgs, s_cfgs) as (s, training, fixed_mapping):
        training_placeholder = training
        # Using chaining writing style:
        logits = globals()[FLAGS.model](x, num_classes, training, weight_decay).tensor

    # Construct the fixed saver
    saver = nf.utils.fixed_model_saver(fixed_mapping)
    
    # Loss and metrics
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy + tf.add_n(reg_losses) if reg_losses else cross_entropy
    index_label = tf.argmax(labels, 1)
    correct = tf.equal(tf.argmax(logits, 1), index_label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    top5_correct = tf.nn.in_top_k(logits, index_label, 5)
    top5_accuracy = tf.reduce_mean(tf.cast(top5_correct, tf.float32))
    
    # Initialize the optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # Learning rate is multiplied by 0.5 after training for every 30 epochs
    learning_rate = tf.train.exponential_decay(0.05, global_step=global_step,
                                               decay_steps=int(x_train.shape[0] / batch_size * 30),
                                               decay_rate=0.5, staircase=True)
    # FIXME: momentum也不能要了...
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    grads_and_vars = optimizer.compute_gradients(loss)
    if FLAGS.prune:
        train_step = nf.apply_gradients_prune(optimizer.apply_gradients, grads_and_vars, 'mask.json', global_step=global_step)
    else:
        train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
 
    # Scales and summary op
    weight_tensor_names, weight_data_scales, weight_grad_scales, \
        weight_data_cfgs, weight_grad_cfgs = zip(*sorted([(k.op.name, v["q_data_scale"], v["q_grad_scale"], v["data_cfg"], v["grad_cfg"])
                                                          for k, v in fixed_mapping[nf.DataTypes.WEIGHT].iteritems()], key=lambda x: x[0]))
    weight_data_names = [t.op.name for t in weight_data_scales]
    weight_grad_names = [t.op.name for t in weight_grad_scales]
    wd_bit_widths = [c.bit_width for c in weight_data_cfgs]
    wg_bit_widths = [c.bit_width for c in weight_grad_cfgs]
    wg_scales_values_min = None
    wg_scales_values_max = None
    # Summary weight data/grad scales
    for t in weight_data_scales:
        ind = t.op.name.index("/data")
        summary_name = t.op.name[:ind].replace("/", "_") + t.op.name[ind:]
        tf.summary.scalar(summary_name, t)
    for t in weight_grad_scales:
        ind = t.op.name.index("/grad")
        summary_name = t.op.name[:ind].replace("/", "_") + t.op.name[ind:]
        tf.summary.scalar(summary_name, t)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    log("Using real-time data augmentation.")
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
    steps_per_epoch = x_train.shape[0] // batch_size
    total_iters = FLAGS.epochs * steps_per_epoch

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.load_from is not None:
            log("Loading checkpoint from {}".format(FLAGS.load_from))
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.load_from))
        else:
            print("error: no input")
            exit(1)

        def accu_func():
            test_gen = datagen_test.flow(x_test, y_test, batch_size=batch_size)
            steps_per_epoch = x_test.shape[0] // batch_size
            results_all = [0,0,0]
            for step in range(1, steps_per_epoch+1):
                x_v, y_v = next(test_gen)
                results = sess.run([loss, accuracy, top5_accuracy],
                                        feed_dict={
                                            x: x_v,
                                            labels: y_v
                                        })
                for i in range(len(results_all)):
                    results_all[i] += results[i]
            print(results_all)
            for i in range(len(results_all)):
                results_all[i] /= float(steps_per_epoch)
            return results_all
        if FLAGS.prune == "manual":
            nf.prune_all(prune_rate, dimenson=[0,0,1,1])
        elif FLAGS.prune == "auto":
            nf.auto_prune_all(accu_func)
        print("Saved model to: ", saver.save(sess, os.path.join(FLAGS.train_dir, "prune")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional file to write logs to.")
    parser.add_argument("--train_dir", type=str, default="",
                        help="Directory for storing snapshots.")
    parser.add_argument("--summary_dir", type=str, default="./summary",
                        help="Directory for storing summarys.")
    parser.add_argument("--model", type=str, default="VGG11",
                        choices=["VGG11", "Conv4Dense2"], help="The network structure.")
    parser.add_argument("--cfg", type=str, default=None,
                        help="The file for fixed configration.")
    parser.add_argument("--scfg", type=str, default=None,
                        help="The file for strategy configration.")
    parser.add_argument("--load-from", type=str, default=None,
                        help="Finetune from the given model checkpoint.")
    parser.add_argument("--save-strategy", type=str, default=None,
                        help="Save the hardware weightgradsaver strategy to this file.")

    parser.add_argument("--test_frequency", type=int, default=5,
                        metavar="N", help="Test the accuracies on validation set "
                        "after every N epochs.")
    parser.add_argument("--weight-grad-iters", type=int, default=100, metavar="WG_ITERS",
                        help="The weight gradient scales of the last WG_ITERS iterations will be used to calculate the gradient buffer scale.")
    parser.add_argument("--grad-buffer-width", type=int, default=24,
                        help="The weight gradient buffer width.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="The max training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The training/testing batch size.")
    parser.add_argument("--prune", type=str, default="auto", choices=["auto", "manual"],
                        help="prune finetune or normal train.")
    parser.add_argument("--gpu", type=str, default="0,1",
                        help="gpu used to train.")

    # FIXME: weight decay是不是不应该加入了
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="The L2 weight decay parameter.")
    FLAGS = parser.parse_args()
    if FLAGS.log_file:
        FLAGS.log_file = open(FLAGS.log_file, "w")
    if not FLAGS.train_dir:
        log("WARNING: model will not be saved if `--train_dir` option is not given.")
    tf.app.run(main=main, argv=[sys.argv[0]])