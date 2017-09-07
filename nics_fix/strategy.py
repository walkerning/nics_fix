# -*- coding: utf-8 -*-

from __future__ import print_function

import re
import abc
import six

import tensorflow as tf
from tensorflow.python.framework.registry import Registry

from nics_fix.logger import logger

__all__ = ["Strategy", "Strategies", "NoiseStrategy"]

def _convert_camel2underscore(camel):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

_strategy_registry = Registry("strategy")

class _StrategyMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, attrs):
        parents = [b for b in bases if isinstance(b, _StrategyMeta)]
        if not parents:
            # BaseStrategy do not need to be registered.
            return super(_StrategyMeta, mcls).__new__(mcls, name, bases, attrs)
        reg_name = _convert_camel2underscore(attrs.get("__registry_name__", name))
        cls = super(_StrategyMeta, mcls).__new__(mcls, name, bases, attrs)
        _strategy_registry.register(cls, reg_name)
        logger.info("Registering strategy {}, implementation class: {}".format(reg_name, name))
        return cls

@six.add_metaclass(_StrategyMeta)
class BaseStrategy(object):
    """
    An abstract strategy interface.
    """
    @abc.abstractmethod
    def pre_weight(self, input):
        pass

    @abc.abstractmethod
    def post_weight(self, input):
        pass

    @abc.abstractmethod
    def pre_weight_grad(self, input):
        pass

    @abc.abstractmethod
    def post_weight_grad(self, input):
        pass

    @abc.abstractmethod
    def pre_activation(self, input):
        pass

    @abc.abstractmethod
    def post_activation(self, input):
        pass

    @abc.abstractmethod
    def pre_activation_grad(self, input):
        pass

    @abc.abstractmethod
    def post_activation_grad(self, input):
        pass

class Strategy(BaseStrategy):
    __registry_name__ = "none_strategy"

    def __init__(self, name, fixed_weight, fixed_activation, cfg):
        pass

    def pre_weight(self, input):
        return input
    
    def post_weight(self, input):
        return input

    def pre_weight_grad(self, input):
        return input

    def post_weight_grad(self, input):
        return input

    def pre_activation(self, input):
        return input

    def post_activation(self, input):
        return input

    def pre_activation_grad(self, input):
        return input

    def post_activation_grad(self, input):
        return input

    @classmethod
    def init_from_cfg(cls, name, fixed_weight, fixed_activation, cfg):
        n_cfg = cfg.copy()
        s_name = n_cfg.pop("name", None)
        if s_name is None:
            raise AssertionError("Require `name` field in every strategy configuration item.")
        s_cls = _strategy_registry.lookup(s_name)
        return s_cls(name, fixed_weight, fixed_activation, **n_cfg)

class Strategies(object):
    def __init__(self, name, strategies):
        self.name = name
        self.strategies = strategies

    def get_func(self, data_type, phase, grad):
        """
        Arguments:
          data_type: "weight" or "activation"
          phase: "pre" or "post"
          grad: True or False
        """
        assert data_type in {"weight", "activation"}
        assert phase in {"pre", "post"}
        assert isinstance(grad, bool)
        func_name = phase + "_" + data_type + ("_grad" if grad else "")
        def func(input, fixed_scale):
            data = input
            if phase == "pre":
                for st in self.strategies:
                    data = getattr(st, func_name)(data, fixed_scale)
            else:
                for st in reversed(self.strategies):
                    data = getattr(st, func_name)(data, fixed_scale)
            return data
        func.func_name = "{}:{}".format(self.name, func_name)
        return func

    @classmethod
    def init_from_cfgs(cls, name, fixed_weight, fixed_activation, cfgs):
        return cls(name, [Strategy.init_from_cfg(name, fixed_weight, fixed_activation, cfg) for cfg in cfgs])

class NoiseStrategy(Strategy):
    """
    NoiseStrategy adds noise to endpoints during the forward/backward pass.
    
    Available strategies are:
      * none: No noise will be added.
      * uniform: Add uniform noise between predefined min/max values.
      * uniform_ratio:  Add uniform noise between min/max values that is relative to the minimal step of quantitization. 
                        Eg. If the binary point is behind the 3rd digits as in: 01101.11, the minimal step will be 0.25,
                            so, if min=-0.5, max=0.5(by default), uniform noise of (-0.125, 0.125) will be added. 
                        If the fixed config of this data is:
                        * "none": This strategy will not do anything in this float-point calculation case, and issue a warning.
                        * "fixed": This strategy will fetch the corresponding fixed scale from FIXED_*_SCALE_COL_KEY collections.
                        * "auto": For now this strategy can only be used on `post_*` methods, and will be handled in the same way
                                  as in the "fixed" situation, because this need calculate the scale based on the data,
                                  and the added noise can potentially change the scale of the next fixed-point phase.

    TODO: The scale tensor of quantitization should be shared to avoid multiple calculations.
          What should we do when adding pre-noise changes the scale.
    """
    __registry_name__ = "noise_strategy"

    _default_data_strategy = {
        "type": "none"
    }
    _default_gradient_strategy = {
        "type": "none"
    }
    _default_uniform_strategy = {
        "min": -0.5,
        "max": 0.5
    }

    def __init__(self, name, fixed_weight, fixed_activation, weight={}, activation={}):
        self.name = name
        self.fixed_weight_cfg = fixed_weight.data_config
        self.fixed_weight_grad_cfg = fixed_weight.gradient_config
        self.fixed_activation_cfg = fixed_activation.data_config
        self.fixed_activation_grad_cfg = fixed_activation.gradient_config

        self.weight_cfg = weight.get("data_config", {})
        self.pre_weight_cfg = self.weight_cfg.get("pre", self._default_data_strategy)
        self.post_weight_cfg = self.weight_cfg.get("post", self._default_data_strategy)

        self.weight_grad_cfg = weight.get("gradient_config", self._default_gradient_strategy)
        self.pre_weight_grad_cfg = self.weight_grad_cfg.get("pre", self._default_gradient_strategy)
        self.post_weight_grad_cfg = self.weight_grad_cfg.get("post", self._default_gradient_strategy)
        
        self.activation_cfg = activation.get("data_config", self._default_data_strategy)
        self.pre_activation_cfg = self.activation_cfg.get("pre", self._default_data_strategy)
        self.post_activation_cfg = self.activation_cfg.get("post", self._default_data_strategy)

        self.activation_grad_cfg = activation.get("gradient_config", self._default_gradient_strategy)
        self.pre_activation_grad_cfg = self.activation_grad_cfg.get("pre", self._default_gradient_strategy)
        self.post_activation_grad_cfg = self.activation_grad_cfg.get("post", self._default_gradient_strategy)
        
        for method_name in ["pre_weight", "post_weight", "pre_weight_grad", "post_weight_grad",
                            "pre_activation", "post_activation", "pre_activation_grad", "post_activation_grad"]:
            cfg = getattr(self,  method_name + "_cfg")
            fixed_cfg = getattr(self, "fixed_" + method_name[method_name.find("_")+1:] + "_cfg")
            setattr(self, method_name, NoiseStrategy.noise_factory(name, method_name, fixed_cfg, cfg))

    @staticmethod
    def noise_factory(name, method_name, fixed_cfg, cfg):
        def func(input, fixed_scale):
            if cfg["type"] == "none":
                return input
            elif cfg["type"] == "uniform":
                t_cfg = NoiseStrategy._default_uniform_strategy.copy()
                t_cfg.update(cfg)
                logger.info("NoiseStrategy:{}:{}: `uniform` strategy(min: {}, max: {}) added.".format(name, method_name, 
                                                                                                     t_cfg["min"], t_cfg["max"]))
                return input + tf.random_uniform(tf.shape(input), t_cfg["min"], t_cfg["max"])
            elif cfg["type"] == "uniform_ratio":
                t_cfg = NoiseStrategy._default_uniform_strategy.copy()
                t_cfg.update(cfg)
                # Only use strategy at training phase
                if fixed_cfg.training is None or fixed_cfg.training == "none":
                    # Issue warning
                    logger.warn(("NoiseStrategy:{}:{}: `uniform_ratio` strategy will have no effect when the fixed configuration"
                                  "is \"none\".").format(name, method_name))
                    return input
                    
                logger.info("NoiseStrategy:{}:{}: `uniform_ratio` strategy(min: {}, max: {}) added.".format(name, method_name,
                                                                                                           t_cfg["min"], t_cfg["max"]))

                if isinstance(fixed_cfg.training, int):
                    step = tf.stop_gradient(tf.pow(2., fixed_cfg.training - (fixed_cfg.bit_width - 1)))
                else:
                    if fixed_cfg.training == "auto" and method_name.startswith("pre_"):
                        # FIXME: Will there be a loop caused by control-dependency?
                        logger.warn(("NoiseStrategy:{}:{}: `uniform_ratio` strategy may have un-intended effect when"
                                      "the fixed configuration is \"auto\", and the applied endpoint is `pre_*`.")\
                                     .format(name, method_name))
                                      
                    step = tf.stop_gradient(tf.pow(2., fixed_scale - (fixed_cfg.bit_width - 1)))
                # For debugging use.
                # input = tf.Print(input, [step, input], "the original input: ", first_n=5, summarize=10)
                output = input + tf.random_uniform(tf.shape(input), t_cfg["min"] * step, t_cfg["max"] * step)
                # output = tf.Print(output, [output], "the added-noise input: ", first_n=5, summarize=10)
                return output
            else:
                raise AssertionError("Available type of NoiseStrategy is: `none`, `uniform`, `uniform_ratio`.")
        func.func_name = method_name
        return func
