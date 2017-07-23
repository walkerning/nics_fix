# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np

from nics_fix.context import get_context, TRAINING_PLACEHOLDER_KEY, FIXED_DATA_COL_KEY,\
    FIXED_GRAD_COL_KEY, FIXED_DATA_SCALE_COL_KEY, FIXED_GRAD_SCALE_COL_KEY

__all__ = ["quantitize"]

def _get_basename(data, full=False):
    last_ind = data.name.rfind("/")
    if full:
        return data.name.split(":")
    else:
        basename = data.name[last_ind + 1:]
        basename, ind = basename.split(":")
        return basename, ind

def _quantitize_data(data, data_fixed_scale, data_cfg, name=None):
    training = get_context(TRAINING_PLACEHOLDER_KEY)
    base_name, _ = _get_basename(data)
    base_name = name if name else base_name
    out_data = tf.where(training,
                        _quantitize_cfg(data, data_fixed_scale, data_cfg.training, data_cfg.bit_width),
                        _quantitize_cfg(data, data_fixed_scale, data_cfg.not_training, data_cfg.bit_width),
                        name="{}_select".format(base_name))
    tf.add_to_collection(FIXED_DATA_COL_KEY, out_data)
    return out_data

def _quantitize_grad(data, grad_fixed_scale, grad_cfg, name=None):
    """
    Arguments:
        `data` is the Tensor get by `_quantitize_data`, its op name is {activation, kernel, bias}_select.
    """
    data_basename, ind = _get_basename(data, full=True)
    gradient_op_name = "QuantGrad_{}:{}".format(data_basename, ind)
    @tf.RegisterGradient(gradient_op_name)
    def _grad(op, output_grad):
        # TODO: Need strategy abstraction to handle this grad. Many ways to handle grad,
        #       like adding random noise, saving grads into a random buffer and so on.
        # Here is the default handling. just call `_quantize_cfg`.
        input_grad = _quantitize_cfg(output_grad, grad_fixed_scale, grad_cfg.config, grad_cfg.bit_width, name=name)
        # Do not need to record the modified gradients here. because the gradients can be found in grads_and_vars directly.
        # TODO: maybe should record `output_grad` for debug use.
        # tf.add_to_collection(FIXED_GRAD_COL_KEY, input_grad)
        return input_grad
    G = tf.get_default_graph()
    # Add fix gradient op
    with G.gradient_override_map({"Identity": gradient_op_name}):
        return tf.identity(data)

def _quantitize_cfg(data, fixed_scale, cfg, bit_width, name=None):
    if cfg is None:
        return tf.identity(data, name=name)
    elif cfg == "auto":
        # Calculate the fixed scale according to all the values and stored
        EPS = 1e-5
        new_scale = tf.ceil(tf.log(tf.reduce_max(tf.abs(data)) + EPS) / np.log(2))
        with tf.control_dependencies([tf.assign(fixed_scale, new_scale)]):
            return _do_quantitize(data, fixed_scale, bit_width, name=name)
    elif cfg == "fixed":
        # Use the last stored fixed scale
        return _do_quantitize(data, fixed_scale, bit_width, name=name)
    else:
        assert isinstance(cfg, int)
        # TODO: Insert intialization op into context.
        return _do_quantitize(data, cfg, bit_width, name=name)

def _do_quantitize(data, scale, bit_width, name):
    step = tf.stop_gradient(tf.pow(2., scale - (bit_width - 1)))
    maximum = tf.stop_gradient(tf.pow(2., scale) - step)
    minimum = tf.stop_gradient(-tf.pow(2., scale))
    G = tf.get_default_graph()
    with G.gradient_override_map({"Round": "Identity"}):
        # TODO: Even if the quantitize cfg is "auto", some overflow may occur, and maybe cause some problems.
        #       such as maybe weights won't be able to be trained to change scale if the learning rate is not big enough.
        # Two possible solutions:
        # * Do not minus step at maximum when training on software, 
        #   this may cause some small discrepancy between software simulation and actual hardware deployment.
        # * Modify the `new_scale` calculation.
        return tf.minimum(tf.maximum(tf.round(data / step) * step, minimum), maximum, name=name)

def quantitize(data, cfg, name=None, scope=None):
    """
    Arguments:
        data: tf.Tensor
        cfg: nics_fix.config.FixedConfig
    """                  
    scope = scope or tf.get_variable_scope()
    with tf.variable_scope(scope) as sc:
        # Very mysterious, in https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/:
        # > When opening a variable scope using a captured object instead of a string, we do not alter the current name scope for ops.
        with tf.name_scope(sc.original_name_scope):
            sc.set_custom_getter(None)
            data_basename, ind = _get_basename(data)
            prefix_name = name if name else "{}_{}".format(data_basename, ind)
            # TODO: Change to use logging module
            print("Quantitze data {} using cfg: {}".format(prefix_name, cfg))
            data_fixed_scale = tf.get_variable(prefix_name + "_data_fixed_scale", shape=(), dtype=tf.float32,
                                               trainable=False, initializer=tf.constant_initializer(0))
            grad_fixed_scale = tf.get_variable(prefix_name + "_grad_fixed_scale", shape=(), dtype=tf.float32,
                                               trainable=False, initializer=tf.constant_initializer(0))
            tf.add_to_collection(FIXED_DATA_SCALE_COL_KEY, data_fixed_scale)
            tf.add_to_collection(FIXED_GRAD_SCALE_COL_KEY, grad_fixed_scale)
    
            return _quantitize_grad(_quantitize_data(data, data_fixed_scale, cfg.data_config, name),
                                    grad_fixed_scale, cfg.gradient_config, name)

                                        
