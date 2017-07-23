# -*- coding: utf-8 -*-

from __future__ import print_function

import yaml

__all__ = ["DataConfig", "GradientConfig", "FixedConfig", "FixedConfigs", "parse_cfg_from_str", "parse_cfg_from_file"]

available_configs = {"auto", "fixed", None}

def _check_legal_config(cfg):
    assert cfg in available_configs or isinstance(cfg, int), "Illegal configuration item: `{}`. Only accept integers or one of {}".format(cfg, available_configs)

class DataConfig(object):
    def __init__(self, training, not_training, bit_width):
        _check_legal_config(training)
        _check_legal_config(not_training)
        assert isinstance(bit_width, int) and bit_width > 0, "Illegal bitwidth: `{}`. Only accept positive integers."
        
        self.bit_width = bit_width
        self.training = training
        self.not_training = not_training

    def __repr__(self):
        return "DataConfig(training={}, not_training={}, bit_width={})".format(self.training, self.not_training, self.bit_width)

class GradientConfig(object):
    def __init__(self, config, bit_width):
        _check_legal_config(config)
        assert isinstance(bit_width, int) and bit_width > 0, "Illegal bitwidth: `{}`. Only accept positive integers."

        self.bit_width = bit_width
        self.config = config

    def __repr__(self):
        return "GradientConfig(config={}, bit_width={})".format(self.config, self.bit_width)

class FixedConfig(object):
    def __init__(self, data_config, gradient_config):
        if not isinstance(data_config, DataConfig):
            data_config = DataConfig(*data_config)
        if not isinstance(gradient_config, GradientConfig):
            gradient_config = DataConfig(gradient_config)
        self.data_config = data_config
        self.gradient_config = gradient_config

    def __repr__(self):
        return "FixedConfig(data_config={}, gradient_config={})".format(self.data_config, self.gradient_config)

_default_weight_config = (DataConfig(training="auto", not_training="fixed", bit_width=8),
                          GradientConfig(None, bit_width=8))
_default_activation_config = (DataConfig(training="auto", not_training="fixed", bit_width=8),
                              GradientConfig(None, bit_width=8))
default_fix_config = (FixedConfig(*_default_weight_config), FixedConfig(*_default_activation_config))

class FixedConfigs(object):
    # `type_default_configs` is the registry of the fixed
    # configuration of each `fixed_registered` fixed op.
    type_default_configs = {}

    def __init__(self, type_configs, name_configs):
        self.type_configs = type_configs
        self.name_configs = name_configs
    
    @classmethod
    def register_default(cls, type_name, config):
        cls.type_default_configs[type_name] = config

    def lookup(self, name, type_name):
        """
        Return:
            A two-element tuple of `FixedConfig`: (weights_config, activation_config).
        """
        # Get fix configuration by type_name of operation
        cfg = self.type_configs.get(type_name, self.type_default_configs[type_name])
        if name is not None and name in self.name_configs:
            # Get fix configuration by the name of operation
            cfg = self.name_configs[name]
        weight_cfg, act_cfg = cfg
        return weight_cfg, act_cfg

    @classmethod
    def parse_cfg_from_str(cls, cfgstr):
        raw_cfg = yaml.load(cfgstr)
        type_configs = {}
        name_configs = {}
        if raw_cfg is not None:
            if "by_type" in raw_cfg:
                type_configs = {type_name: (FixedConfig(*type_cfg.get("weight", _default_weight_config)), FixedConfig(*type_cfg.get("activation", _default_activation_config))) for type_name, type_cfg in raw_cfg["by_type"].iteritems()}
            if "by_name" in raw_cfgs:
                name_configs = {op_name: (FixedConfig(*op_cfg.get("weight", _default_weight_config)), FixedConfig(*op_cfg.get("activation", _default_activation_config))) for op_name, op_cfg in raw_cfg["by_type"].iteritems()}
        return cls(type_configs, name_configs)

    @classmethod
    def parse_cfg_from_file(cls, fname):
        with open(fname, "r") as f:
            return cls.parse_cfg_from_str(f.read())


parse_cfg_from_str = FixedConfigs.parse_cfg_from_str
parse_cfg_from_file = FixedConfigs.parse_cfg_from_file



