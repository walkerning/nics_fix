# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging

__all__ = ["logger"]

log_level = os.environ.get("NICS_FIX_LOGLEVEL", "INFO").upper()
log_level = getattr(logging, log_level, logging.INFO)
print("NICS_FIX: Use log level: {}".format(logging.getLevelName(log_level)))
logger = logging.getLogger("NICS_FIX")
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s: [%(levelname)s] %(asctime)s %(module)8s:%(lineno)2d] %(message)s"))
logger.addHandler(handler)
