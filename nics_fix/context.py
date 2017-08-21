# -*- coding: utf-8 -*-

from __future__ import print_function

from contextlib import contextmanager

import tensorflow as tf
#from tensorflow.python.util import tf_contextlib

from nics_fix.consts import DataTypes
from nics_fix.config import FixedConfigs
from nics_fix.strategy_config import StrategyConfigs

__all__ = ["get_context", "fixed_scope", "FIXED_MAPPING_KEY"]

# Context used by fixed scope
_context = {}

FIXED_CONFIG_KEY = "__fixed_config"
FIXED_STRATEGY_CONFIG_KEY = "__fixed_strategy_config"
FIXED_MAPPING_KEY = "__fixed_mapping"
TRAINING_PLACEHOLDER_KEY = "fix_training_placeholder"

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
