# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps

import tensorflow as tf

from nics_fix.consts import FixedKeys

__all__ = ["fixed_model_saver"]

def fixed_model_saver(fixed_mapping, fixed_weight=False):
    saver_dct = {tensor.op.name: tensor for tensor in tf.global_variables()}
    tensor_tmp_var = {}

    if fixed_weight:
        # construct temporary variables
        weight_fixed_mapping = fixed_mapping.get("weight", {})
        with tf.variable_scope("A_SAFE_NEVER_WILL_OCCUR_SCOPE"):
            tensor_tmp_var = {tensor.op.name: (info["q_data"], tf.get_variable(tensor.op.name, shape=info["q_data"].get_shape())) \
                                               for tensor, info in weight_fixed_mapping.iteritems()}
        weight_dct = {name: tmp_var for name, (tmp_var, _) in tensor_tmp_var.iteritems()}
        saver.update(weight_dct)
    saver = tf.train.Saver(saver_dct)
    
    @wraps(tf.train.Saver.save)
    def _patch_save(self, sess, *args, **kwargs):
        if tensor_tmp_var:
            sess.run([tf.assign(tmp_var, q_data) for q_data, tmp_var in tensor_tmp_var.itervalues()])
        ret_v = tf.train.Saver.save(self, sess, *args, **kwargs)
        return ret_v

    saver.save = _patch_save.__get__(saver, tf.train.Saver)
    return saver
    
