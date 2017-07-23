# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps, partial

import tensorflow as tf
from tensorflow.python.framework.registry import Registry

from nics_fix.config import FixedConfigs, default_fix_config
from nics_fix.context import get_context, FIXED_CONFIG_KEY
from nics_fix.quant import quantitize


__all__ = ["fixed_register", "wrap"]

fixed_ops_registry = Registry("fixed ops")

class _Holder(tf.Tensor):
    def __init__(self, tensor):
        self._tensor = tensor

    def __getattr__(self, name):
        if name in fixed_ops_registry.list():
            return partial(fixed_ops_registry.lookup(name), self._tensor)
        # Default, proxy to inner tensor
        return getattr(self._tensor, name)

    @property
    def tensor(self):
        return self._tensor

def wrap(tensor):
    return _Holder(tensor)

def map_variables(func):
    def _custom_getter(getter, *args, **kwargs):
        return func(getter(*args, **kwargs))
    return _custom_getter

def fixed_register(inner_func, type_name, default_config=default_fix_config):
    @wraps(inner_func)
    def _true_func(*args, **kwargs):
        cfg = get_context(FIXED_CONFIG_KEY)
        if cfg is None:
            raise RuntimeError("You can not use the registered fixed operations outside the fixed context created by `fixed_scope`.")
        name = kwargs.get("name", None)
        weight_cfg, act_cfg = cfg.lookup(name, type_name)
        cur_scope = tf.get_variable_scope()
        # FIXME: I don't know why. The name of layers cannot be uniqued automatically within this context... As the build of Tensorflow takes time, I will debug this issue later on.
        kwargs["name"] = tf.get_default_graph().unique_name(name if name else type_name)
        kwargs["name"] = kwargs["name"][kwargs["name"].rfind("/", 1) + 1:]
        with tf.variable_scope(cur_scope, custom_getter=map_variables(partial(quantitize, cfg=weight_cfg))):
            res = inner_func(*args, **kwargs)
        res = _Holder(quantitize(res, act_cfg, scope=kwargs.get("name", None), name="activation"))
        return res
    # Register this fixed op
    fixed_ops_registry.register(_true_func, type_name)
    FixedConfigs.register_default(type_name, default_config)
    return _true_func

fixed_register(tf.layers.conv2d, "Conv2d")
fixed_register(tf.layers.dense, "Dense")
fixed_register(tf.nn.relu, "ReLU")
