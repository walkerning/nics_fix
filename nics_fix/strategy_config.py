# -*- coding: utf-8 -*-

from __future__ import print_function

import yaml

__all__ = ["StrategyConfigs", "parse_strategy_cfg_from_str", "parse_strategy_cfg_from_file"]

class StrategyConfigs(object):
    def __init__(self, type_configs, name_configs):
        self.type_configs = type_configs
        self.name_configs = name_configs
    
    @classmethod
    def register_default(cls, type_name, config):
        cls.type_default_configs[type_name] = config

    def lookup(self, name, type_name, default=None):
        """
        Return:
            A list of dict.
        """
        # Get fix configuration by type_name of operation
        cfg = self.type_configs.get(type_name, default)
        if name is not None and name in self.name_configs:
            # Get fix configuration by the name of operation
            cfg = self.name_configs[name]
        if (not cfg is default) and (not isinstance(cfg, list)):
            assert isinstance(cfg, dict)
            cfg = [cfg]
        return cfg

    @classmethod
    def parse_cfg_from_str(cls, cfgstr):
        raw_cfg = yaml.load(cfgstr)
        type_configs = {}
        name_configs = {}
        if raw_cfg is not None:
            if "by_type" in raw_cfg:
                type_configs = raw_cfg["by_type"]
            if "by_name" in raw_cfg:
                name_configs = raw_cfg["by_name"]
        return cls(type_configs, name_configs)

    @classmethod
    def parse_cfg_from_file(cls, fname):
        with open(fname, "r") as f:
            return cls.parse_cfg_from_str(f.read())


parse_strategy_cfg_from_str = StrategyConfigs.parse_cfg_from_str
parse_strategy_cfg_from_file = StrategyConfigs.parse_cfg_from_file
