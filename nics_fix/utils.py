# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps

import tensorflow as tf

import numpy as np

import random

from nics_fix.consts import FixedKeys

__all__ = ["fixed_model_saver", "RandomCrop"]

def fixed_model_saver(fixed_mapping, fixed_weight=False, *args, **kwargs):
    saver_dct = {tensor.op.name: tensor for tensor in tf.global_variables()}
    tensor_tmp_var = {}

    if fixed_weight:
        # construct temporary variables
        weight_fixed_mapping = fixed_mapping.get("weight", {})
        with tf.variable_scope("A_SAFE_NEVER_WILL_OCCUR_SCOPE"):
            tensor_tmp_var = {tensor.op.name: (info["q_data"], tf.get_variable(tensor.op.name, shape=info["q_data"].get_shape())) \
                                               for tensor, info in weight_fixed_mapping.iteritems()}
        weight_dct = {name: tmp_var for name, (_, tmp_var) in tensor_tmp_var.iteritems()}
        saver_dct.update(weight_dct)
    saver = tf.train.Saver(saver_dct, *args, **kwargs)
    
    @wraps(tf.train.Saver.save)
    def _patch_save(self, sess, *args, **kwargs):
        if tensor_tmp_var:
            sess.run([tf.assign(tmp_var, q_data) for q_data, tmp_var in tensor_tmp_var.itervalues()])
        ret_v = tf.train.Saver.save(self, sess, *args, **kwargs)
        return ret_v

    saver.save = _patch_save.__get__(saver, tf.train.Saver)
    return saver
    
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