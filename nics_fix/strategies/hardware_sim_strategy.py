# -*- coding: utf-8 -*-

from __future__ import print_function

from nics_fix.quant import _do_quantitize
from nics_fix.strategy import Strategy
from nics_fix.logger import logger

class WeightGradSaverStrategy(Strategy):
    """
    This strategy assume the gradient saving buffer has a fixed scale range.
    Any gradient accumulation that is outside this range will be cut.
    """
    __registry_name__ = "weightgradsaver_strategy"

    def __init__(self, name, fixed_weight, fixed_activation, scale=0, bit_width=24):
        self.name = name
        self.scale = scale
        self.bit_width = bit_width

    def post_weight_grad(self, input, sc):
        logger.info("{}: post weight grad to {}. scale={}, bit_width={}.".format(self.__class__.__name__,
                                                                                 input.op.name, self.scale, self.bit_width))
        return _do_quantitize(input, self.scale, self.bit_width, None)
        
