# -*- coding: utf-8 -*-

from __future__ import print_function

import logging

import tensorflow as tf
import numpy as np

from nics_fix.context import get_context, TRAINING_PLACEHOLDER_KEY
from nics_fix.consts import FixedKeys, DataTypes, _get_fixed_key
from nics_fix.strategy import Strategies

__all__ = ["quantitize"]

def _get_basename(data, full=False):
    last_ind = data.name.rfind("/")
    if full:
        return data.name.split(":")
    else:
        basename = data.name[last_ind + 1:]
        basename, ind = basename.split(":")
        return basename, ind

def _quantitize_data(data, data_fixed_scale, data_cfg, name=None, pre_data=None, post_data=None, col_key=None):
    training = get_context(TRAINING_PLACEHOLDER_KEY)
    base_name, _ = _get_basename(data)
    base_name = name if name else base_name

    # Must do this for using `pre/post_data` in the lambda expression below.
    if pre_data is None:
        pre_data = lambda d, s: d
    if post_data is None:
        post_data = lambda d, s: d

    # tf.control_flow_ops.cond: conditional execution applies only to the operations defined in `true_fn` and `false_fn`
    out_data = tf.cond(training,
                       lambda: post_data(_quantitize_cfg(pre_data(data, data_fixed_scale), data_fixed_scale,
                                                         data_cfg.training, data_cfg.bit_width), data_fixed_scale),
                       lambda: _quantitize_cfg(data, data_fixed_scale, data_cfg.not_training, data_cfg.bit_width),
                       name="{}_select".format(base_name))

    if col_key:
        tf.add_to_collection(col_key, out_data)
    return out_data

def _quantitize_grad(data, grad_fixed_scale, grad_cfg, name=None, pre_grad=None, post_grad=None):
    """
    Arguments:
        `data` is the Tensor get by `_quantitize_data`, its op name is {activation, kernel, bias}_select.
    """
    data_basename, ind = _get_basename(data, full=True)
    gradient_op_name = "QuantGrad_{}:{}".format(data_basename, ind)
    @tf.RegisterGradient(gradient_op_name)
    def _grad(op, output_grad):
        if pre_grad is not None:
            output_grad = pre_grad(output_grad, grad_fixed_scale)
        # TODO: Need strategy abstraction to handle this grad. Many ways to handle grad,
        #       like adding random noise, saving grads into a random buffer and so on.
        # Here is the default handling. just call `_quantize_cfg`.
        input_grad = _quantitize_cfg(output_grad, grad_fixed_scale, grad_cfg.training, grad_cfg.bit_width, name=name)
        # Do not need to record the modified gradients here. because the gradients can be found in grads_and_vars directly.
        # TODO: maybe should record `output_grad` for debug use.
        # tf.add_to_collection(FIXED_GRAD_COL_KEY, input_grad)
        if post_grad is not None:
            input_grad = post_grad(input_grad, grad_fixed_scale)
        return input_grad
    G = tf.get_default_graph()
    # Add fix gradient op
    with G.gradient_override_map({"Identity": gradient_op_name}):
        return tf.identity(data)

def _quantitize_cfg(data, fixed_scale, cfg, bit_width, name=None):
    if cfg is None or cfg == "none":
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
        #       Or, I think maybe we do not need to do this initialization, due to the following use case:
        #       Maybe we have some quantitized model, but we need to test it at another scale,
        #       and we do not want to coincidently override the saved fixed scales.
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

def quantitize(data, cfg, name=None, scope=None, strategies=None, data_type=DataTypes.OTHER, col_key_prefix=None):
    """
    Arguments:
        data: tf.Tensor
        cfg: nics_fix.config.FixedConfig
    """
    assert strategies is None or isinstance(strategies, Strategies)
    assert data_type in DataTypes.all

    scope = scope or tf.get_variable_scope()
    with tf.variable_scope(scope) as sc:
        # Very mysterious, in https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/:
        # > When opening a variable scope using a captured object instead of a string,
        # > we do not alter the current name scope for ops.
        with tf.name_scope(sc.original_name_scope):
            sc.set_custom_getter(None)
            data_basename, ind = _get_basename(data)
            prefix_name = "{}_{}".format(name if name else data_basename, ind)
            logging.info("Quantitze data {} using cfg: {}".format(prefix_name, cfg))
            with tf.variable_scope("fixed_scale"):
                # data_fixed_scale = tf.get_variable(prefix_name + "_data_fixed_scale", shape=(), dtype=tf.float32,
                #                                    trainable=False, initializer=tf.constant_initializer(0))
                # grad_fixed_scale = tf.get_variable(prefix_name + "_grad_fixed_scale", shape=(), dtype=tf.float32,
                #                                    trainable=False, initializer=tf.constant_initializer(0))
                data_fixed_scale = tf.get_variable("data/" + prefix_name, shape=(), dtype=tf.float32,
                                                   trainable=False, initializer=tf.constant_initializer(0))
                grad_fixed_scale = tf.get_variable("grad/" + prefix_name, shape=(), dtype=tf.float32,
                                                   trainable=False, initializer=tf.constant_initializer(0))

                data_fixed_col_key = col_key_prefix + "_data_scale" if col_key_prefix else\
                                     _get_fixed_key(data_type, scale=True)
                grad_fixed_col_key = col_key_prefix + "_grad_scale" if col_key_prefix else\
                                     _get_fixed_key(data_type, grad=True, scale=True)

                if data_fixed_col_key is not None:
                    tf.add_to_collection(data_fixed_col_key, data_fixed_scale)
                if grad_fixed_col_key is not None:
                    tf.add_to_collection(grad_fixed_col_key, grad_fixed_scale)

            if strategies is not None:
                pre_data = strategies.get_func(data_type=data_type, phase="pre", grad=False)
                post_data = strategies.get_func(data_type=data_type, phase="post", grad=False)
                pre_grad = strategies.get_func(data_type=data_type, phase="pre", grad=True)
                post_grad = strategies.get_func(data_type=data_type, phase="post", grad=True)
            else:
                pre_data = None
                post_data = None
                pre_grad = None
                post_grad = None

            data_col_key = col_key_prefix + "_data" if col_key_prefix else _get_fixed_key(data_type)
            return _quantitize_grad(_quantitize_data(data, data_fixed_scale, cfg.data_config, name,
                                                     pre_data=pre_data, post_data=post_data, col_key=data_col_key),
                                    grad_fixed_scale, cfg.gradient_config, name, pre_grad=pre_grad, post_grad=post_grad)
                                    

