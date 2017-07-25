# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging

log_level = os.environ.get("NICS_FIX_LOGLEVEL", "WARNING").upper()
log_level = getattr(logging, log_level, logging.WARNING)
print("NICS_FIX: Use log level: {}".format(logging.getLevelName(log_level)))
logging.basicConfig(
    level=log_level,
    format="NICS_FIX: [%(levelname)s] %(asctime)s %(module)8s:%(lineno)2d] %(message)s"
)

from .context import *
from .ops import *
from .ops_inner import *
from .quant import *
from .config import *
from .strategy import *
from .strategy_config import *
