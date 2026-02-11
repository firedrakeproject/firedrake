# Copyright (c) 2026, Imperial College London and others.
# Please see the AUTHORS file in the main source directory for
# a full list of copyright holders. All rights reserved.

"""The PyOP2 logger, based on the Python standard library logging module."""

from contextlib import contextmanager
import logging


LOGGER = logging.getLogger('pyop3')

debug = LOGGER.debug
info = LOGGER.info
warning = LOGGER.warning
error = LOGGER.error
critical = LOGGER.critical

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def log(level, msg, *args, **kwargs):
    ''' Print 'msg % args' with the severity 'level'.

    :arg level: the log level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    :arg msg: the message '''

    LOGGER.log(level, msg, *args, **kwargs)


_indent = 0


@contextmanager
def progress(level, msg, *args, **kwargs):
    """A context manager to print a progress message.

    The block is wrapped in ``msg...``, ``msg...done`` log messages
    with an appropriate indent (to distinguish nested message).

    :arg level: the log level.  See :func:`log` for valid values
    :arg msg: the message.

    See :func:`log` for more details.
    """
    global _indent
    log(level, (' ' * _indent) + msg + '...', *args, **kwargs)
    _indent += 2
    yield
    _indent -= 2
    log(level, (' ' * _indent) + msg + '...done', *args, **kwargs)
