
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
# Ensure that the relevant loggers have been created.
import tsfc.logging             # noqa: F401
import pyop2.logger             # noqa: F401

from pyop2.configuration import configuration
from pyop2.mpi import COMM_WORLD


__all__ = ('set_level', 'set_log_level', 'set_log_handlers',
           'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'debug', 'info', 'warning', 'error', 'critical',
           'info_red', 'info_green', 'info_blue',
           "RED", "GREEN", "BLUE")


packages = ("pyop2", "tsfc", "firedrake", "UFL")


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


def set_log_handlers(handlers=None, comm=COMM_WORLD):
    """Set handlers for the log messages of the different Firedrake components.

    :kwarg handlers: Optional dict of handlers keyed by the name of the logger.
         If not provided, a separate :class:`logging.StreamHandler`
         will be created for each logger.
    :kwarg comm: The communicator the handler should be collective
         over.  If provided, only rank-0 on that communicator will
         write to the handler, other ranks will use a
         :class:`logging.NullHandler`.  If set to ``None``, all ranks
         will use the provided handler.  This could be used, for
         example, if you want to log to one file per rank.
    """
    if handlers is None:
        handlers = {}

    for package in packages:
        logger = logging.getLogger(package)
        for handler in logger.handlers:
            logger.removeHandler(handler)

        handler = handlers.get(package, None)
        if handler is None:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))

        if comm is not None and comm.rank != 0 and not configuration["spmd_strict"]:
            handler = logging.NullHandler()

        logger.addHandler(handler)


def set_log_level(level):
    """Set the log level for Firedrake components.

    :arg level: The level to use.

    This controls what level of logging messages are printed to
    stderr.  The higher the level, the fewer the number of messages.

    """
    for package in packages:
        logger = logging.getLogger(package)
        logger.setLevel(level)


set_level = set_log_level
