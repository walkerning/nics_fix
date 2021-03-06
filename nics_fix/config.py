# -*- coding: utf-8 -*-

from __future__ import print_function

import yaml

__all__ = ["DataConfig", "GradientConfig", "FixedConfig", "FixedConfigs", "parse_cfg_from_str", "parse_cfg_from_file"]

available_configs = {"auto", "fixed", "none", None}

def _check_legal_config(cfg):
    assert cfg in available_configs or isinstance(cfg, int), "Illegal configuration item: `{}`. Only accept integers or one of {}".format(cfg, available_configs)

# Do we all need default values for each configuration?
class DataConfig(object):
    # def __init__(self, training="auto", not_training="fixed", bit_width=8):
    def __init__(self, training="none", not_training="none", bit_width=8):
        _check_legal_config(training)
        _check_legal_config(not_training)
        assert isinstance(bit_width, int) and bit_width > 0, "Illegal bitwidth: `{}`. Only accept positive integers."
        
        self.bit_width = bit_width
        self.training = training
        self.not_training = not_training

    def __repr__(self):
        return "DataConfig(training={}, not_training={}, bit_width={})".format(self.training, self.not_training, self.bit_width)

    def __eq__(self, another):
        return isinstance(another, DataConfig) and self.bit_width == another.bit_width and self.training == another.training and self.not_training == another.not_training

class GradientConfig(object):
    def __init__(self, training=None, bit_width=32):
        _check_legal_config(training)
        assert isinstance(bit_width, int) and bit_width > 0, "Illegal bitwidth: `{}`. Only accept positive integers."

        self.bit_width = bit_width
        self.training = training

    def __repr__(self):
        return "GradientConfig(config={}, bit_width={})".format(self.training, self.bit_width)

    def __eq__(self, another):
        return isinstance(another, GradientConfig) and self.bit_width == another.bit_width and self.training == another.training

# _default_data_config = DataConfig(training="auto", not_training="fixed", bit_width=8)
_default_data_config = DataConfig(training="none", not_training="none")
_default_gradient_config = GradientConfig(None, bit_width=32)
_default_weight_config = {
    "data_config": _default_data_config,
    "gradient_config": _default_gradient_config
}
_default_activation_config = {
    "data_config": _default_data_config,
    "gradient_config": _default_gradient_config
}

class FixedConfig(object):
    def __init__(self, data_config=_default_data_config, gradient_config=_default_gradient_config):
        if not isinstance(data_config, DataConfig):
            data_config = DataConfig(**data_config)
        if not isinstance(gradient_config, GradientConfig):
            gradient_config = GradientConfig(**gradient_config)
        self.data_config = data_config
        self.gradient_config = gradient_config

    def __repr__(self):
        return "FixedConfig(data_config={}, gradient_config={})".format(self.data_config, self.gradient_config)

    def __eq__(self, another):
        return isinstance(another, FixedConfig) and self.data_config == another.data_config and self.gradient_config == another.gradient_config

default_fix_config = (FixedConfig(**_default_weight_config), FixedConfig(**_default_activation_config))

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
    def _merge_cfg(cls, cfg, available_cfgs):
        # TODO: need recursive udpate?
        fix_cfg = {}
        if "config_name" in cfg:
            fix_cfg.update(available_cfgs[cfg["config_name"]])
        fix_cfg.update(cfg)
        return fix_cfg

    @classmethod
    def parse_cfg_from_str(cls, cfgstr):
        raw_cfg = yaml.load(cfgstr)
        type_configs = {}
        name_configs = {}
        if raw_cfg is not None:
            if "configs" in raw_cfg:
                available_cfgs = raw_cfg["configs"]
            if "by_type" in raw_cfg:
                type_configs = {}
                for type_name, type_cfg in raw_cfg["by_type"].iteritems():
                    type_cfg = cls._merge_cfg(type_cfg, available_cfgs)
                    type_configs[type_name] = (FixedConfig(**type_cfg.get("weight", _default_weight_config)), FixedConfig(**type_cfg.get("activation", _default_activation_config)))
            if "by_name" in raw_cfg:
                name_configs = {}
                for op_name, op_cfg in raw_cfg["by_name"].iteritems():
                    op_cfg = cls._merge_cfg(op_cfg, available_cfgs)
                    name_configs[op_name] = (FixedConfig(**op_cfg.get("weight", _default_weight_config)), FixedConfig(**op_cfg.get("activation", _default_activation_config)))
        return cls(type_configs, name_configs)

    @classmethod
    def parse_cfg_from_file(cls, fname):
        with open(fname, "r") as f:
            return cls.parse_cfg_from_str(f.read())


parse_cfg_from_str = FixedConfigs.parse_cfg_from_str
parse_cfg_from_file = FixedConfigs.parse_cfg_from_file

FixedConfigs.register_default("data", default_fix_config)
