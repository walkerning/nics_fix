# -*- coding: utf-8 -*-

import pytest
from nics_fix.config import *
from nics_fix.config import _default_activation_config, _default_weight_config, _default_gradient_config

test_config_cases = [
    {
        "cfg_str": """
by_type:
  Conv2d:
    weight:
      data_config: 
        bit_width: 4
        not_training: fixed
        training: auto
      gradient_config:
        bit_width: 8
        training: auto
by_name:
  conv_1:
    activation:
      data_config:
        bit_width: 8
        training: 2
""",
        "lookups": {
            (None, "Conv2d"): (FixedConfig(DataConfig("auto", "fixed", 4), GradientConfig("auto", 8)),
                               FixedConfig(**_default_activation_config)),
            ("conv_1", "Conv2d"): (FixedConfig(**_default_weight_config),
                                   FixedConfig(DataConfig(2, "fixed", 8),
                                               _default_gradient_config))
        }
    }
 ]

@pytest.mark.parametrize("case", test_config_cases)
def test_practice1(case):
    cfg = parse_cfg_from_str(case["cfg_str"])
    for look, ans in case["lookups"].iteritems():
        assert cfg.lookup(*look) == ans
