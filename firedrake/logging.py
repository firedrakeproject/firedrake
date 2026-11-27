
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
# Ensure that the TSFC logger has been created.
import tsfc.logging             # noqa: F401

import pyop3.config
from pyop3.log import ANSI_BLUE, ANSI_GREEN, ANSI_RED, ANSI_RESET
from pyop3.mpi import COMM_WORLD


__all__ = ('set_level', 'set_log_level', 'set_log_handlers',
           'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'debug', 'info', 'warning', 'error', 'critical',
           'info_red', 'info_green', 'info_blue',
           "RED", "GREEN", "BLUE")


packages = ("pyop3", "tsfc", "firedrake", "UFL")


logger = logging.getLogger("firedrake")
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

RED = f"{ANSI_RED}%s{ANSI_RESET}"
BLUE = f"{ANSI_BLUE}%s{ANSI_RESET}"
GREEN = f"{ANSI_GREEN}%s{ANSI_RESET}"


# Mimic dolfin
def info_red(message, *args, **kwargs):
    """Log an INFO message in red.

    Parameters
    ----------
    message : str
        The log message.
    *args : object
        Arguments that the logger substitutes into the message.
    **kwargs : object
        Keyword arguments for :meth:`logging.Logger.info`.
    """
    info(f"{ANSI_RED}{message}{ANSI_RESET}", *args, **kwargs)


def info_green(message, *args, **kwargs):
    """Log an INFO message in green.

    Parameters
    ----------
    message : str
        The log message.
    *args : object
        Arguments that the logger substitutes into the message.
    **kwargs : object
        Keyword arguments for :meth:`logging.Logger.info`.
    """
    info(f"{ANSI_GREEN}{message}{ANSI_RESET}", *args, **kwargs)


def info_blue(message, *args, **kwargs):
    """Log an INFO message in blue.

    Parameters
    ----------
    message : str
        The log message.
    *args : object
        Arguments that the logger substitutes into the message.
    **kwargs : object
        Keyword arguments for :meth:`logging.Logger.info`.
    """
    info(f"{ANSI_BLUE}{message}{ANSI_RESET}", *args, **kwargs)


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
        # Avoids duplicating messages if someone directly invokes logging.info:
        logger.propagate = False

        for handler in logger.handlers:
            logger.removeHandler(handler)

        handler = handlers.get(package, None)
        if handler is None:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))

        if comm is not None and comm.rank != 0 and not pyop3.config.spmd_strict:
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
