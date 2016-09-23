"""Logging for TSFC."""

from __future__ import absolute_import, print_function, division

import logging

logger = logging.getLogger('tsfc')
logger.addHandler(logging.StreamHandler())
