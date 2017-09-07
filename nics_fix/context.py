# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import inspect
from contextlib import contextmanager

import tensorflow as tf
#from tensorflow.python.util import tf_contextlib

from nics_fix.consts import DataTypes
from nics_fix.config import FixedConfigs
from nics_fix.strategy_config import StrategyConfigs

__all__ = ["get_context", "fixed_scope", "kwargs_scope", "kwargs_scope_by_type", "FIXED_MAPPING_KEY"]

# Context used by fixed scope
_context = {}

FIXED_CONFIG_KEY = "__fixed_config"
FIXED_STRATEGY_CONFIG_KEY = "__fixed_strategy_config"
FIXED_MAPPING_KEY = "__fixed_mapping"
TRAINING_PLACEHOLDER_KEY = "fix_training_placeholder"
METHOD_KWARGS_KEY = "__method_kwargs"

def get_context(key):
    return _context.get(key, None)

#@tf_contextlib.contextmanager
@contextmanager
def fixed_scope(name_or_scope, fixed_config, strategy_config=None):
    """
    Create a fixed scope context.
    An example of using:
    ```python
    import nics_fix as nf
    x = tf.placeholder(tf.float32, shape=(None, 784))
    with nf.fixed_scope("fixed", parse_cfg_from_file("config.yaml")):
        res = nf.wrap(x).Dense(units=10).tensor
    
    ```
    """
    if not isinstance(fixed_config, FixedConfigs):
        raise TypeError("`fixed_config` must be an instance of FixedConfigs.")
    if strategy_config is not None and not isinstance(strategy_config, StrategyConfigs):
        raise TypeError("`strategy_config` must be None or an instance of StrategyConfigs.")
    assert FIXED_CONFIG_KEY not in _context, "Nested fixed scope not allowed."
    try:
        # Add a training placeholder
        # training = tf.placeholder(tf.bool, name=TRAINING_PLACEHOLDER_KEY)
        training = tf.placeholder_with_default(False, shape=(),
                                               name=TRAINING_PLACEHOLDER_KEY)
        fixed_mapping = {name: {} for name in DataTypes.all}

        _context[TRAINING_PLACEHOLDER_KEY] = training
        _context[FIXED_CONFIG_KEY] = fixed_config
        _context[FIXED_STRATEGY_CONFIG_KEY] = strategy_config
        _context[FIXED_MAPPING_KEY] = fixed_mapping
        with tf.variable_scope(name_or_scope) as scope:
            yield scope, training, fixed_mapping
    finally:
        _context.pop(FIXED_CONFIG_KEY, None)
        _context.pop(FIXED_STRATEGY_CONFIG_KEY, None)
        _context.pop(FIXED_MAPPING_KEY, None)
        _context.pop(TRAINING_PLACEHOLDER_KEY, None)

@contextmanager
def kwargs_scope(_override=False, **kwargs):
    """
    Created a default keyword arguments context.

    An example of using:
    ```python
    with kwargs_scope(**{"Conv2D.kernel_size": (3, 3),
                         "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=scale)}):
        logits = nf.wraps(x).Conv2D(name="conv1", kernel_size=(7, 7), filters=32)\
                 .Conv2D(name="conv2", filters=32).Flatten()
                 .Dense(units=10, name="dense3").tensor
    ```
    As both `tf.layers.conv2d` and `tf.layers.dense` receive keyword arguments named `kernel_regularizer`,
    this default regularization keyword arguments will be applied to all the three layers created (conv1, conv2, dense3).
    The layer "conv2" will have kernel size (3, 3), whereas layer "conv1" will have kernel size (7, 7), 
    overriding the default value (3, 3).
    """
    old_kwargs = copy.copy(_context.get(METHOD_KWARGS_KEY, None))
    if _override or METHOD_KWARGS_KEY not in _context:
        _context[METHOD_KWARGS_KEY] = copy.copy(kwargs)
    else:
        _context[METHOD_KWARGS_KEY].update(kwargs)
    yield
    if old_kwargs is None:
        _context.pop(METHOD_KWARGS_KEY, None)
    else:
        _context[METHOD_KWARGS_KEY] = old_kwargs

@contextmanager
def kwargs_scope_by_type(_override=False, **kwargs):
    """
    Created a default keyword arguments context. Specified by type names.
    All the value of specifications should be a dictionary of format `{argname: argvalue}`.

    An example of using:
    ```python
    with kwargs_scope(Conv2D={"kernel_size": (3, 3),
                              "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=scale1)},
                      MaxPool={"strides": 2,
                               "pool_size": (2, 2)},
                      Dense={"kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=scale2)})
        logits = nf.wraps(x).Conv2D(name="conv1", kernel_size=(7, 7), filters=32)\
                 .MaxPool(name="pool1")
                 .Conv2D(name="conv2", filters=64)
                 .MaxPool(name="pool2")
                 .Conv2D(name="conv3", filters=128)
                 .Flatten()
                 .Dense(units=10, name="dense5").tensor
    ```
    The three convolution layers will have the same kernel regularizer scale, layer "conv1" have kernel size (7, 7), whereas
    other two convolution layers have kernel size (3, 3). All max pooling layers will have `strides` equals to 2, `pool_size`
    equals to (2, 2). The dense layer use a different kernel regularization scale with convolution layers.
    """
    assert all(isinstance(v, dict) for v in kwargs.values()), \
        "`kwargs_scope_by_type` receive kwargs only in this format: {'typename1' : {'argname1': argvalue1, 'argname2': argvalue2} ...}"
    old_kwargs = copy.copy(_context.get(METHOD_KWARGS_KEY, None))
    qualified_kwargs = {".".join([type_name, argname]): argvalue for type_name, argspec in kwargs.iteritems() for argname, argvalue in argspec.iteritems()}
    if _override or METHOD_KWARGS_KEY not in _context:
        _context[METHOD_KWARGS_KEY] = qualified_kwargs
    else:
        _context[METHOD_KWARGS_KEY].update(qualified_kwargs)
    yield
    if old_kwargs is None:
        _context.pop(METHOD_KWARGS_KEY, None)
    else:
        _context[METHOD_KWARGS_KEY] = old_kwargs

def get_kwargs(func, type_name, true_kwargs):
    default_kwargs = get_context(METHOD_KWARGS_KEY)
    if not default_kwargs:
        return true_kwargs
    arg_names = set(inspect.getargspec(func).args)
    func_default_kwargs = {argname.split(".")[-1] : argvalue for argname, argvalue in default_kwargs.iteritems()
                           if "." not in argname or argname.startswith(type_name + ".")}
    kwargs = {argname : argvalue for argname, argvalue in func_default_kwargs.iteritems()
              if argname in arg_names}
    kwargs.update(true_kwargs)
    return kwargs
