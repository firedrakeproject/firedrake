"""Logging for TSFC."""

import logging

logger = logging.getLogger('tsfc')
logger.addHandler(logging.StreamHandler())
