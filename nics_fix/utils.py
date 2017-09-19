# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps

import numpy as np
import tensorflow as tf

from nics_fix.consts import FixedKeys, DataTypes
from nics_fix.logger import logger

__all__ = ["fixed_model_saver", "amend_weight_grad_scales"]

def fixed_model_saver(fixed_mapping, fixed_weight=False, **kwargs):
    saver_dct = {tensor.op.name: tensor for tensor in tf.global_variables()}
    tensor_tmp_var = {}

    if fixed_weight:
        # construct temporary variables
        weight_fixed_mapping = fixed_mapping.get("weight", {})
        with tf.variable_scope("A_SAFE_NEVER_WILL_OCCUR_SCOPE"):
            tensor_tmp_var = {tensor.op.name: (info["q_data"], tf.get_variable(tensor.op.name,
                                                                               shape=info["q_data"].get_shape())) \
                                               for tensor, info in weight_fixed_mapping.iteritems()}
        weight_dct = {name: tmp_var for name, (_, tmp_var) in tensor_tmp_var.iteritems()}
        saver_dct.update(weight_dct)

    saver = tf.train.Saver(saver_dct, **kwargs)
    
    @wraps(tf.train.Saver.save)
    def _patch_save(self, sess, *args, **kwargs):
        if tensor_tmp_var:
            sess.run([tf.assign(tmp_var, q_data) for q_data, tmp_var in tensor_tmp_var.itervalues()])
        ret_v = tf.train.Saver.save(self, sess, *args, **kwargs)
        return ret_v

    saver.save = _patch_save.__get__(saver, tf.train.Saver)
    return saver

def amend_weight_grad_scales(learning_rate, wg_scales_values_min, wg_scales_values_max,
                             weight_data_scales, weight_grad_scales,
                             weight_data_bit_widths, weight_grad_bit_widths,
                             from_min=True, grad_buffer_width=None,
                             sess=None):
    """
    Arguments
    ------------
    fixed_mapping: The fixed mapping returned by `nf.fixed_scope` context manager.
                   Used to lookup bit width configurations.
    learning_rate: The current learning rate value or the learning rate tensor.
                   If `learning_rate` is a tf.Tensor,
                   will use the sess(default to the default session) to get the current learning_rate value.
    wg_scales_values_min: The weight gradient fixed scales of each weight tensor.
                          Will be used if `from_min` == True.
    wg_scales_values_max: The weight gradient fixed scales of each weight tensor.
                          Will be used if `from_min` == False.
    weight_data_scales: The weight data fixed scales tensors.
    weight_grad_scales: The weight gradient fixed scales tensors. Do not use here...
    weight_data_bit_widths: A list of the data bit width of weight tensors.
    weight_grad_bit_widths: A list of the gradient bit width of weight tensors.
    from_min: If `True`, the gradient scale will be amend using `wg_scales_values_min`,
              `grad_buffer_width` will not be used.
              If `False`, the gradient scale will be amend using `wg_scales_values_max`,
              `grad_buffer_width` must be supplied.
    grad_buffer_width: The width of the gradient accumulation buffer.
    sess: The tensorflow session to use.

    Returns
    ------------
    grad_scales_values: List of int.
                        The amended gradient scales,
                        should be updated to `weight_grad_scales` tensors outside this function before saving.
    grad_buffer_scales_values: List of int.
                               The fixed scales of gradient accumulation buffer.
    """
    if sess is None:
        sess = sess or tf.get_default_session()
    if from_min:
        assert grad_buffer_width is not None

    # Get data scales of every weight tensor
    weight_data_scales_values = sess.run(weight_data_scales)

    # Get the learning rate shift
    lr = learning_rate
    if isinstance(learning_rate, tf.Tensor):
        lr = sess.run(learning_rate)
    shift = np.log2(lr)
    EPS = 1e-5
    hardware_shift = np.floor(shift)
    hardware_lr = np.exp2(hardware_shift)
    if shift - np.round(shift) > EPS:
        logger.warn(("The current learning rate {} is not a power of 2, this value is not friendly to hardware."
                     " Will be coverted to floor({}) = {} (shift=).").format(lr, lr, hardware_lr, hardware_shift))
    logger.info("LR: {}; SHIFT: {}".format(hardware_lr, hardware_shift))

    grad_scales_values = []
    grad_buffer_scales_values = []
    for ds, dw, gw, wg_min, wg_max in zip(weight_data_scales_values, weight_data_bit_widths, weight_grad_bit_widths,
                                          wg_scales_values_min, wg_scales_values_max):
        gs_v, gbs_v = _get_weight_grad_buffer_scale(hardware_shift, ds, dw, gw,
                                                    grad_buffer_width, wg_min, wg_max)
        grad_scales_values.append(gs_v)
        grad_buffer_scales_values.append(gbs_v)
    return grad_scales_values, grad_buffer_scales_values
    
def _get_weight_grad_buffer_scale(lr_shift, data_scale, data_width, grad_width, grad_buffer_width,
                                  min_grad_scale, max_grad_scale, from_min=True):
    """
    Return:
    scale: The fixed scale of the weight gradient buffer.
    NOTE: The actual fixed scale of which that is used to do update will be `scale + lr_shift`
          after multiplying learning rate.
    """
    if not from_min:
        return max_grad_scale, max(data_scale - (data_width - 1) - lr_shift, max_grad_scale)
    else:
        return min_grad_scale, max(data_scale - (data_width - 1) - lr_shift,
                                   min_grad_scale - (grad_width - 1) + grad_buffer_width)
