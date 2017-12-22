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

from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, BasicIterativeMethod
from cleverhans.attacks_tf import jacobian_graph

import tensorflow as tf
import nics_fix as nf

FLAGS = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class QCNN(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, cfgs):
        super(Model, self).__init__()
        self.cfgs = cfgs
        self.logits = None
        self.fixed_mapping = None
        self.last_x = None
        self.training_status = None
        self.mid = None

    def get_training_status(self):
        return self.training_status

    def get_mid(self):
        return self.mid

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        weight_decay = FLAGS.weight_decay
        num_classes = 10
        if self.logits != None and self.last_x == x:
            return self.logits
        self.last_x = x
        if self.logits != None:
            reuse = True
        else:
            reuse = False
        with nf.fixed_scope("fixed_mlp_mnist", self.cfgs, reuse=reuse) as (s, training, fixed_mapping):
            self.training_status = training
            self.fixed_mapping = fixed_mapping
            with nf.kwargs_scope_by_type(Conv2D={"padding": "same",
                                         "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                         "kernel_initializer": tf.contrib.layers.variance_scaling_initializer()},
                                 MaxPool={"pool_size": (2, 2),
                                          "strides": 2},
                                 BN={"is_training": self.training_status, "scale": True},
                                 Dense={
                                     "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=weight_decay)}):
                if FLAGS.net == "resnet20":
                    conv1 = nf.Conv2D(x, filters=16, kernel_size=(3,3), name="conv1")
                    bn1 = nf.BN(conv1, scope="bn1")
                    relu1 = nf.ReLU(bn1, name="relu1")
                    conv2 = nf.Conv2D(relu1, filters=16, kernel_size=(3,3), name="conv2")
                    bn2 = nf.BN(conv2, scope="bn2")
                    relu2 = nf.ReLU(bn2, name="relu2")
                    conv3 = nf.Conv2D(relu2, filters=16, kernel_size=(3,3), name="conv3")
                    bn3 = nf.BN(conv3, scope="bn3")
                    elt1 = bn3 + relu1
                    relu3 = nf.ReLU(elt1, name="rule3")
                    conv4 = nf.Conv2D(relu3, filters=16, kernel_size=(3,3), name="conv4")
                    bn4 = nf.BN(conv4, scope="bn4")
                    relu4 = nf.ReLU(bn4, name="relu4")
                    conv5 = nf.Conv2D(relu4, filters=16, kernel_size=(3,3), name="conv5")
                    bn5 = nf.BN(conv5, scope="bn5")
                    elt2 = bn5 + relu3
                    relu5 = nf.ReLU(elt2, name="rule5")
                    conv6 = nf.Conv2D(relu5, filters=16, kernel_size=(3,3), name="conv6")
                    bn6 = nf.BN(conv6, scope="bn6")
                    relu6 = nf.ReLU(bn6, name="relu6")
                    conv7 = nf.Conv2D(relu6, filters=16, kernel_size=(3,3), name="conv7")
                    bn7 = nf.BN(conv7, scope="bn7")
                    elt3 = bn7 + relu5
                    relu7 = nf.ReLU(elt3, name="rule7")
                    conv8 = nf.Conv2D(relu7, filters=32, kernel_size=(1,1), strides=(2,2), name="conv8")
                    bn8 = nf.BN(conv8, scope="bn8")
                    conv9 = nf.Conv2D(relu7, filters=32, kernel_size=(3,3), strides=(2,2), name="conv9")
                    bn9 = nf.BN(conv9, scope="bn9")
                    relu8 = nf.ReLU(bn9, name="relu8")
                    conv10 = nf.Conv2D(relu8, filters=32, kernel_size=(3,3), name="conv10")
                    bn10 = nf.BN(conv10, scope="bn10")
                    elt4 = bn8 + bn10
                    relu9 = nf.ReLU(elt4, name="relu9")
                    conv11 = nf.Conv2D(relu9, filters=32, kernel_size=(3,3), name="conv11")
                    bn11 = nf.BN(conv11, scope="bn11")
                    relu11 = nf.ReLU(bn11, name="relu11")
                    conv12 = nf.Conv2D(relu11, filters=32, kernel_size=(3,3), name="conv12")
                    bn12 = nf.BN(conv12, scope="bn12")
                    elt5 = bn12 + relu9
                    relu11 = nf.ReLU(elt5, name="rule11")
                    conv13 = nf.Conv2D(relu11, filters=32, kernel_size=(3,3), name="conv13")
                    bn13 = nf.BN(conv13, scope="bn13")
                    relu12 = nf.ReLU(bn12, name="relu12")
                    conv14 = nf.Conv2D(relu12, filters=32, kernel_size=(3,3), name="conv14")
                    bn14 = nf.BN(conv14, scope="bn14")
                    elt6 = bn14 + relu11
                    relu13 = nf.ReLU(elt6, name="rule13")
                    conv15 = nf.Conv2D(relu13, filters=64, kernel_size=(1,1), strides=(2,2), name="conv15")
                    bn15 = nf.BN(conv15, scope="bn15")
                    conv16 = nf.Conv2D(relu13, filters=64, kernel_size=(3,3), strides=(2,2), name="conv16")
                    bn16 = nf.BN(conv16, scope="bn16")
                    relu14 = nf.ReLU(bn16, name="relu14")
                    conv17 = nf.Conv2D(relu14, filters=64, kernel_size=(3,3), name="conv17")
                    bn17 = nf.BN(conv17, scope="bn17")
                    elt7 = bn15 + bn17
                    relu15 = nf.ReLU(elt7, name="relu15")
                    conv18 = nf.Conv2D(relu15, filters=64, kernel_size=(3,3), name="conv18")
                    bn18 = nf.BN(conv18, scope="bn18")
                    relu16 = nf.ReLU(bn16, name="relu16")
                    conv19 = nf.Conv2D(relu16, filters=64, kernel_size=(3,3), name="conv19")
                    bn19 = nf.BN(conv19, scope="bn19")
                    elt8 = bn19 + relu15
                    relu17 = nf.ReLU(elt8, name="relu17")
                    conv20 = nf.Conv2D(relu17, filters=64, kernel_size=(3,3), name="conv20")
                    bn20 = nf.BN(conv20, scope="bn20")
                    relu18 = nf.ReLU(bn20, name="relu18")
                    conv21 = nf.Conv2D(relu18, filters=64, kernel_size=(3,3), name="conv21")
                    bn21 = nf.BN(conv21, scope="bn21")
                    elt9 = bn21 + relu17
                    relu19 = nf.ReLU(elt9, name="relu19")
                    pool1 = nf.AvePool(relu19, pool_size=(8,8), strides=(8,8), name="pool1")
                    flat = nf.Flatten(pool1)
                    ip1 = nf.Dense(flat, units=10, name="ip1",
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                    self.logits = ip1
                    self.mid = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9,
                                conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17,
                                conv18, conv19, conv20, conv21]
                    return self.logits
                elif FLAGS.net == "vgg11":
                    conv1 = (nf.wrap(x).Conv2D(filters=64, kernel_size=(3,3), name="conv1"))
                    conv2 = (conv1.ReLU(name="relu1")
                        .MaxPool(name="pool1")
                        .Conv2D(filters=128, kernel_size=(3,3), name="conv2"))
                    conv3_1 = (conv2
                        .ReLU(name="relu2")
                        .MaxPool(name="pool2")
                        .Conv2D(filters=256, kernel_size=(3,3), name="conv3_1"))
                    conv3_2 = (conv3_1
                        .ReLU(name="relu3_1")
                        .Conv2D(filters=256, kernel_size=(3,3), name="conv3_2"))
                    conv4_1 = (conv3_2
                        .ReLU(name="relu3_2")
                        .MaxPool(name="pool3")
                        .Conv2D(filters=512, kernel_size=(3,3), name="conv4_1"))
                    conv4_2 = (conv4_1
                        .ReLU(name="relu4_1")
                        .Conv2D(filters=512, kernel_size=(3,3), name="conv4_2"))
                    conv5_1 = (conv4_2
                        .ReLU(name="relu4_2")
                        .MaxPool(name="pool4")
                        .Conv2D(filters=512, kernel_size=(3,3), name="conv5_1"))
                    conv5_2 = (conv5_1
                        .ReLU(name="relu5_1")
                        .Conv2D(filters=512, kernel_size=(3,3), name="conv5_2"))
                    self.logits = (conv5_2
                        .ReLU(name="relu5_2")
                        .MaxPool(name="pool5")
                        .Flatten()
                        .Dense(units=512, name="dense6",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                        .ReLU(name="relu6")
                        .Dense(units=512, name="dense7",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                        .ReLU(name="relu7")
                        .Dense(units=num_classes, name="dense8",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())).tensor
                    self.mid = [conv1, conv2, conv3_1, conv3_2
                    , conv4_1, conv4_2, conv5_1, conv5_2]
                    return self.logits
                else:
                    return None

    def get_fixed_mapping(self):
        return self.fixed_mapping

    def get_probs(self, x):
        return tf.nn.softmax(self.get_logits(x))


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
    batch_size = FLAGS.batch_size # default to 128
    num_classes = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    log(x_train.shape[0], " train samples")
    log(x_test.shape[0], " test samples")
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if FLAGS.net == "resnet20":
        x_train -= 128
        x_test -= 128
    x_train /= 255
    x_test /= 255
    
    cfgs = nf.parse_cfg_from_str("")
    if FLAGS.cfg is not None:
        cfgs = nf.parse_cfg_from_file(FLAGS.cfg)

    x = tf.placeholder(tf.float32, shape=[None] + list(x_train.shape[1:]))
    labels = tf.placeholder(tf.float32, [None, num_classes])

    model = QCNN(cfgs)
    logits = model.get_logits(x)
    training = model.get_training_status()
    mid = model.get_mid()
    
    # Loss and metrics
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = cross_entropy + tf.add_n(reg_losses)
    #grads, = tf.gradients(loss, x)
    index_label = tf.argmax(labels, 1)
    correct = tf.equal(tf.argmax(logits, 1), index_label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    top5_correct = tf.nn.in_top_k(logits, index_label, 5)
    top5_accuracy = tf.reduce_mean(tf.cast(top5_correct, tf.float32))

    # Initialize the optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    if FLAGS.net == "vgg11":
        learning_rate = tf.train.exponential_decay(FLAGS.start_lr, global_step=global_step,
                                                   decay_steps=int(x_train.shape[0] / batch_size * 30),
                                                   decay_rate=FLAGS.lr_decay, staircase=True)
    elif FLAGS.net == "resnet20":
        boundaries = [64000, 96000]
        values = [FLAGS.start_lr, FLAGS.start_lr * FLAGS.lr_decay, FLAGS.start_lr * FLAGS.lr_decay * FLAGS.lr_decay]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        exit(1)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    if FLAGS.net == "resnet20":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_var = optimizer.compute_gradients(loss)
            train_step = optimizer.apply_gradients(grads_and_var, global_step=global_step)
    else:
        train_step = optimizer.minimize(loss, global_step=global_step)    
    
    log("Using real-time data augmentation.")
    # This will do preprocessing and realtime data augmentation:
    preprocessing_switch = (FLAGS.net == "vgg11")
    datagen_train = ImageDataGenerator(
        featurewise_center=preprocessing_switch,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=preprocessing_switch,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen_test = ImageDataGenerator(
        featurewise_center=preprocessing_switch,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=preprocessing_switch,  # divide inputs by std of the dataset
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
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if FLAGS.attack == "fgsm":
        attack_params = {'eps': FLAGS.param_one}
    elif FLAGS.attack == "bim":
        attack_params = {'eps': FLAGS.param_one,
                       'eps_iter': FLAGS.param_two}
    elif FLAGS.attack == "jsma":
        attack_params = {'theta': FLAGS.param_one, 'gamma': FLAGS.param_two,
                       'y_target': None}
    elif FLAGS.attack == "cw":
        attack_params = {'binary_search_steps': 1,
                 "y": None,
                 'max_iterations': 100,
                 'learning_rate': 0.1,
                 'batch_size': 100,
                 'initial_const': 10}
    elif FLAGS.attack == "none":
        pass
    else:
        print("error: unknown method.")
        exit(1)
    global_acc = 0
    with tf.Session(config=config) as sess:
        if FLAGS.attack == "fgsm":
            attack_method = FastGradientMethod(model, sess=sess)
            adv_x = attack_method.generate(x, **attack_params)
        elif FLAGS.attack == "bim":
            attack_method = BasicIterativeMethod(model, sess=sess)
            adv_x = attack_method.generate(x, **attack_params)
        elif FLAGS.attack == "jsma":
            attack_method = SaliencyMapMethod(model, sess=sess)
        elif FLAGS.attack == "cw":
            attack_method = CarliniWagnerL2(model, sess=sess)
        elif FLAGS.attack == "none":
            adv_x = x
        saver = nf.utils.fixed_model_saver(model.get_fixed_mapping(), max_to_keep=30)
        sess.run(tf.global_variables_initializer())
        log("Start training...")
        def test_net():
            test_gen = MultiProcessGen(datagen_test.flow(x_test, y_test, batch_size=batch_size))
            steps_per_epoch = x_test.shape[0] // batch_size
            loss_test = 0
            acc_1_test = 0
            acc_5_test = 0
            try:
                for step in range(1, steps_per_epoch+1):
                    x_v, y_v = next(test_gen)
                    loss_v, acc_1, acc_5 = sess.run([loss, accuracy, top5_accuracy],
                                            feed_dict={
                                                x: x_v,
                                                labels: y_v,
                                                training: False
                                            })
                    print("\r\ttest steps: {}/{}".format(step, steps_per_epoch), end="")
                    loss_test += loss_v
                    acc_1_test += acc_1
                    acc_5_test += acc_5
                loss_test /= steps_per_epoch
                acc_1_test /= steps_per_epoch
                acc_5_test /= steps_per_epoch
                log("\r\tTest: loss: {}; top1 accuracy: {:.2f} %; top5 accuracy: {:2f} %.".format(loss_test, acc_1_test * 100, acc_5_test * 100), flush=True)
            finally:
                test_gen.stop()
            return acc_1_test

        def train_net():
            global_acc = 0
            gen = MultiProcessGen(datagen_train.flow(x_train, y_train, batch_size=batch_size))
            try:
                for epoch in range(1, FLAGS.epochs+1):
                    start_time = time.time()
                    steps_per_epoch = x_train.shape[0] // batch_size
                    loss_v_epoch = 0
                    acc_1_epoch = 0
                    acc_5_epoch = 0

                    # Train batches
                    for step in range(1, steps_per_epoch+1):
                        # TODO: use another thread to execute the data augumentation and enqueue
                        x_v, y_v = next(gen)
                        x_crop_v = random_crop(x_v)
                        _, loss_v, acc_1, acc_5, learning_rate_ = sess.run([train_step, loss, accuracy, top5_accuracy, learning_rate],
                                                           feed_dict={
                                                               x: x_crop_v,
                                                               labels: y_v,
                                                               training: True
                                                           })
                        print("\rEpoch {}: steps {}/{} loss: {}".format(epoch, step, steps_per_epoch, loss_v), end="")
                        loss_v_epoch += loss_v
                        acc_1_epoch += acc_1
                        acc_5_epoch += acc_5

                    loss_v_epoch /= steps_per_epoch
                    acc_1_epoch /= steps_per_epoch
                    acc_5_epoch /= steps_per_epoch

                    duration = time.time() - start_time
                    sec_per_batch = duration / (steps_per_epoch * batch_size)
                    log("\r{}: Epoch {}; (average) loss: {:.3f}; (average) top1 accuracy: {:.2f} %; (average) top5 accuracy: {:.2f} %. learning rate: {:.4f}. {:.3f} sec/batch"\
                          .format(datetime.now(), epoch, loss_v_epoch, acc_1_epoch * 100, acc_5_epoch * 100, learning_rate_, sec_per_batch), flush=True)
                    # End training batches

                    # Test on the validation set
                    if epoch % FLAGS.test_frequency == 0:
                        acc_1_test = test_net()

                        if global_acc < acc_1_test and FLAGS.train_dir:
                            global_acc = acc_1_test
                            if not os.path.exists(FLAGS.train_dir):
                                subprocess.check_call("mkdir -p {}".format(FLAGS.train_dir),
                                                      shell=True)
                            log("Saved model to: ", saver.save(sess, FLAGS.train_dir + str(acc_1_test), global_step=global_step))


                    # End test on the validation set
                # End training
            finally:
                gen.stop()

            if FLAGS.train_dir:
                if not os.path.exists(FLAGS.train_dir):
                    subprocess.check_call("mkdir -p {}".format(FLAGS.train_dir),
                                          shell=True)
                log("Saved model to: ", saver.save(sess, FLAGS.train_dir))
        #log(tf.get_default_graph().get_collection("global_variables"))
        if FLAGS.finetune:
            saver.restore(sess, FLAGS.finetune)
            test_net()
        # Training
        if not FLAGS.load_file:
            train_net()
        else:
            saver.restore(sess, FLAGS.train_dir + FLAGS.load_file)
            if FLAGS.prune != "":
                def accu():
                    return [test_net()]
                nf.auto_prune_all(accu, mask_file=FLAGS.prune)
                with tf.control_dependencies(update_ops):
                    grads_and_var = optimizer.compute_gradients(loss)
                    train_step = nf.apply_gradients_prune(optimizer.apply_gradients, grads_and_var, FLAGS.prune, global_step=global_step)
                train_net()
            else:
                test_net()
        if FLAGS.attack in ["fgsm", "bim"]:
            logits_adv = model.get_logits(adv_x)
            training_adv = model.get_training_status()
            mid_adv = model.get_mid()

            # Loss and metrics
            cross_entropy_adv = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_adv))
            reg_losses_adv = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_adv = cross_entropy_adv + tf.add_n(reg_losses_adv)
            correct_adv = tf.equal(tf.argmax(logits_adv, 1), index_label)
            accuracy_adv = tf.reduce_mean(tf.cast(correct_adv, tf.float32))
            top5_correct_adv = tf.nn.in_top_k(logits_adv, index_label, 5)
            top5_accuracy_adv = tf.reduce_mean(tf.cast(top5_correct_adv, tf.float32))
            steps_per_epoch = x_test.shape[0] // batch_size
        elif FLAGS.attack == "jsma":
            batch_size = 1
            steps_per_epoch = 1000
        elif FLAGS.attack == "cw":
            batch_size = 1000
            steps_per_epoch = 1
        else:
            return
        test_gen = MultiProcessGen(datagen_test.flow(x_test, y_test, batch_size=batch_size, shuffle=False))
        loss_test = 0
        acc_1_test = 0
        acc_5_test = 0
        try:
            for step in range(1, steps_per_epoch+1):
                x_v, y_v = next(test_gen)
                if FLAGS.attack in ["jsma", "cw"]:
                    if FLAGS.attack == "jsma":
                        loss_v, acc_1, acc_5, logits_  = sess.run([loss, accuracy, top5_accuracy, logits],
                                        feed_dict={
                                            x: x_v,
                                            labels: y_v,
                                            training: False
                                        })
                        if acc_1 < 1:
                            continue
                        one_hot_target = np.zeros((1, 10), dtype=np.float32)
                        one_hot_target[0, logits_.argmin()] = 1
                        attack_params["y_target"] = one_hot_target
                        adv_x = attack_method.generate_np(x_v, **attack_params)
                        loss_v, acc_1, acc_5  = sess.run([loss, accuracy, top5_accuracy],
                                        feed_dict={
                                            x: adv_x,
                                            labels: y_v,
                                            training: False
                                        })
                        loss_test += loss_v
                        acc_1_test += acc_1
                        acc_5_test += acc_5
                    else:
                        adv_x = attack_method.generate_np(x_v, **attack_params)
                        for i in range(10):
                            loss_v, acc_1, acc_5  = sess.run([loss, accuracy, top5_accuracy],
                                        feed_dict={
                                            x: adv_x[i*100:(i+1)*100],
                                            labels: y_v[i*100:(i+1)*100],
                                            training: False
                                        })
                            loss_test += loss_v
                            acc_1_test += acc_1
                            acc_5_test += acc_5
                        loss_test /= 10
                        acc_1_test /= 10
                        acc_5_test /= 10
                else:
                    loss_v, acc_1, acc_5, mid_ = sess.run([loss, accuracy, top5_accuracy, mid],
                                        feed_dict={
                                            x: x_v,
                                            labels: y_v,
                                            training: False
                                        })
                    if FLAGS.load_data != "":
                        loss_v_adv, acc_1_adv, acc_5_adv, mid_adv_ = sess.run([loss, accuracy, top5_accuracy, mid],
                                            feed_dict={
                                                x: np.load(FLAGS.load_data),
                                                labels: y_v,
                                                training: False
                                            })
                        print("acc_1_adv:", acc_1_adv)
                        exit(1)
                    else:
                        loss_v_adv, acc_1_adv, acc_5_adv, mid_adv_, adv_x_ = sess.run([loss_adv, accuracy_adv, top5_accuracy_adv, mid_adv, adv_x],
                                            feed_dict={
                                                x: x_v,
                                                labels: y_v,
                                                training_adv: False
                                            })
                    if FLAGS.generate_data != "":
                        np.save(FLAGS.generate_data, adv_x_)
                        print("finish generate data")
                        test_gen.stop()
                        exit(1)
                    if FLAGS.save_data != "":
                        print("acc_1:", acc_1)
                        print("acc_1_adv:", acc_1_adv)
                        if not os.path.exists(FLAGS.save_data):
                            subprocess.check_call("mkdir -p {}".format(FLAGS.save_data),
                                                  shell=True)
                        assert(len(mid_) == len(mid_adv_))
                        for i in range(len(mid_)):
                            np.save(FLAGS.save_data + str(2*i) + ".npy", mid_[i].flatten())
                            np.save(FLAGS.save_data + str(2*i+1) + ".npy", mid_adv_[i].flatten())
                        print("finish save data")
                        test_gen.stop()
                        exit(1)
                    print("\r\ttest steps: {}/{}".format(step, steps_per_epoch), end="")
                    loss_test += loss_v_adv
                    acc_1_test += acc_1_adv
                    acc_5_test += acc_5_adv
            loss_test /= steps_per_epoch
            acc_1_test /= steps_per_epoch
            acc_5_test /= steps_per_epoch
            log("\r\tAttack Test: loss: {}; top1 accuracy: {:.2f} %; top5 accuracy: {:2f} %.".format(loss_test, acc_1_test * 100, acc_5_test * 100), flush=True)
        finally:
            test_gen.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional file to write logs to.")
    parser.add_argument("--cfg", type=str, default=None,
                        help="The file for fixed configration.")
    parser.add_argument("--train_dir", type=str, default="",
                        help="Directory for storing snapshots")
    parser.add_argument("--load_file", type=str, default="")
    parser.add_argument("--test_frequency", type=int, default=5,
                        metavar="N", help="Test the accuracies on validation set "
                        "after every N epochs.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="The max training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The training/testing batch size.")
    parser.add_argument("--param_one", type=float, default=0.01)
    parser.add_argument("--param_two", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--start_lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--attack", type=str, default="none")
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--save_data", type=str, default="")
    parser.add_argument("--generate_data", type=str, default="")
    parser.add_argument("--load_data", type=str, default="")
    parser.add_argument("--prune", type=str, default="")
    parser.add_argument("--net", type=str, default="resnet20", choices=["vgg11", "resnet20"])
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.log_file:
        FLAGS.log_file = open(FLAGS.log_file, "w")
    if not FLAGS.train_dir:
        log("WARNING: model will not be saved if `--train_dir` option is not given.")
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

