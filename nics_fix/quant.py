# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np

from nics_fix.logger import logger
from nics_fix.context import get_context, TRAINING_PLACEHOLDER_KEY, FIXED_MAPPING_KEY
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

def _quantitize_data(data, data_fixed_scale, data_cfg, name=None,
                     pre_data=None, post_data=None, col_key=None,
                     data_type=None, data_ori=None):
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
    if data_type and data_ori is not None:
        fixed_mapping = get_context(FIXED_MAPPING_KEY)[data_type]
        fixed_mapping.setdefault(data_ori, {})["q_data"] = out_data
        fixed_mapping[data_ori]["data_cfg"] = data_cfg
    return out_data

def _quantitize_grad(data, grad_fixed_scale, grad_cfg, name=None, pre_grad=None, post_grad=None,
                     data_type=None, data_ori=None):
    """
    Arguments:
        `data` is the Tensor get by `_quantitize_data`, its op name is {activation, kernel, bias}_select.
    """
    data_basename, ind = _get_basename(data, full=True)
    gradient_op_name = "QuantGrad_{}:{}".format(data_basename, ind)
    if data_type and data_ori is not None:
        fixed_mapping = get_context(FIXED_MAPPING_KEY)[data_type]
    else:
        fixed_mapping = None

    @tf.RegisterGradient(gradient_op_name)
    def _grad(op, output_grad):
        if pre_grad is not None:
            output_grad = pre_grad(output_grad, grad_fixed_scale)
        input_grad = _quantitize_cfg(output_grad, grad_fixed_scale, grad_cfg.training, grad_cfg.bit_width, name=name)
        # Do not need to record the modified gradients here. because the gradients can be found in `grads_and_vars` directly.
        # TODO: maybe should record `output_grad` for debug use.
        # tf.add_to_collection(FIXED_GRAD_COL_KEY, input_grad)
        if post_grad is not None:
            input_grad = post_grad(input_grad, grad_fixed_scale)
        if fixed_mapping is not None:
            fixed_mapping.setdefault(data_ori, {})["ori_grad"] = output_grad
            fixed_mapping[data_ori]["grad_cfg"] = grad_cfg
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
        # Users are expected to see the correct fixed point scale
        # in the scales collection, no matter what config this data is used.
        # Init assignment op will be not fetchable here because of the control 
        # dependency, also just initialize once will cause problems 
        # when use different durint training and not-training phase,
        # so do the assiginment everytime the quantitization is done.
        with tf.control_dependencies([tf.assign(fixed_scale, cfg)]):
            return _do_quantitize(data, fixed_scale, bit_width, name=name)

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
            logger.info("Quantitze data {} using cfg: {}".format(prefix_name, cfg))
            with tf.variable_scope("fixed_scale"):
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
                fixed_mapping = get_context(FIXED_MAPPING_KEY)[data_type]
                fixed_mapping.setdefault(data, {})["q_data_scale"] = data_fixed_scale
                fixed_mapping[data]["q_grad_scale"] = grad_fixed_scale

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

            # FIXME: Maybe we do not need the collections anymore as we have the fixed mapping in the context already...
            data_col_key = col_key_prefix + "_data" if col_key_prefix else _get_fixed_key(data_type)
            return _quantitize_grad(_quantitize_data(data, data_fixed_scale, cfg.data_config, name,
                                                     pre_data=pre_data, post_data=post_data, col_key=data_col_key,
                                                     data_type=data_type, data_ori=data),
                                    grad_fixed_scale, cfg.gradient_config, name, pre_grad=pre_grad, post_grad=post_grad,
                                    data_type=data_type, data_ori=data)
                                    

