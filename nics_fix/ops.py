# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
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

    def apply(self, func):
        return _Holder(func(self._tensor))

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

def _recursive_eval(lst, func, type_check=None):
    if isinstance(lst, (list, tuple)):
        return [_recursive_eval(item, func, type_check) for item in lst]
    if type_check:
        if not isinstance(lst, type_check):
            raise TypeError("Type check failed in `_recursive_eval`: expected type {}, got type {}"\
                            .format(type_check, type(lst)))
    return func(lst)

def fixed_register(inner_func, type_name, default_config=default_fix_config):
    @wraps(inner_func)
    def _true_func(*args, **kwargs):
        cfg = get_context(FIXED_CONFIG_KEY)
        if cfg is None:
            raise RuntimeError("You can not use the registered fixed operations outside the fixed context created by `fixed_scope`.")
        name = kwargs.get("name", None)
        weight_cfg, act_cfg = cfg.lookup(name, type_name)
        cur_scope = tf.get_variable_scope()
        # FIXME: I don't know why. The name of layers cannot be uniqued automatically within this context...
        # As a temporary fix, I call unique_name manually here. 
        # As the build of Tensorflow takes time, I will debug this issue later.
        # And call this `unique_name` will result in `Dense` and `Dense_2`, expected `Dense` and `Dense_1`... To be debug...
        scope_name = tf.get_default_graph().unique_name(name if name else type_name)
        scope_name = scope_name[scope_name.rfind("/", 1)+1:]
        with tf.variable_scope(scope_name) as s:
            logging.debug("scope name: {}, original name scope: {}".format(s.name, s.original_name_scope))
            #with tf.variable_scope(cur_scope) as s, tf.name_scope(s.original_name_scope):
            custom_getter = map_variables(partial(quantitize, cfg=weight_cfg, name=name, scope=s))
            s.set_custom_getter(custom_getter)
            res = inner_func(*args, **kwargs)
            s.set_custom_getter(None)
            res = _recursive_eval(res, lambda x: _Holder(quantitize(x, act_cfg, name="activation")), type_check=tf.Tensor)
        return res

    # Register this fixed op
    fixed_ops_registry.register(_true_func, type_name)
    FixedConfigs.register_default(type_name, default_config)
    return _true_func
