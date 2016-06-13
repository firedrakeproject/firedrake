from __future__ import absolute_import

import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
# Ensure that the relevant loggers have been created.
import tsfc.logging
import pyop2.logger
import coffee.logger
from ufl.log import ufl_logger

from pyop2.mpi import COMM_WORLD


__all__ = ('set_level', 'set_log_level', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'debug', 'info', 'warning', 'error', 'critical',
           'info_red', 'info_green', 'info_blue',
           "RED", "GREEN", "BLUE")


packages = ("COFFEE", "pyop2", "tsfc", "firedrake", "UFL")


for package in packages:
    if package != "UFL":
        logger = logging.getLogger(package)
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Only print on rank 0.
    if COMM_WORLD.rank == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))
    else:
        handler = logging.NullHandler()

    if package == "UFL":
        ufl_logger.set_handler(handler)
    else:
        logger.addHandler(handler)

logger = logging.getLogger("firedrake")
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"


# Mimic dolfin
def info_red(message, *args, **kwargs):
    ''' Write info message in red.

    :arg message: the message to be printed. '''
    info(RED % message, *args, **kwargs)


def info_green(message, *args, **kwargs):
    ''' Write info message in green.

    :arg message: the message to be printed. '''
    info(GREEN % message, *args, **kwargs)


def info_blue(message, *args, **kwargs):
    ''' Write info message in blue.

    :arg message: the message to be printed. '''
    info(BLUE % message, *args, **kwargs)


def set_log_level(level):
    """Set the log level for Firedrake components.

    :arg level: The level to use.

    This controls what level of logging messages are printed to
    stderr.  The higher the level, the fewer the number of messages.

    """
    for package in packages:
        if package == "UFL":
            from ufl.log import ufl_logger as logger
            logger.set_level(level)
        else:
            logger = logging.getLogger(package)
            logger.setLevel(level)


set_level = set_log_level
