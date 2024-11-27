# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""The PyOP2 logger, based on the Python standard library logging module."""

from contextlib import contextmanager
import logging

logger = logging.getLogger('pyop2')
handler = logging.StreamHandler()
logger.addHandler(handler)


debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def set_log_level(level):
    '''Set the log level of the PyOP2 logger.

    :arg level: the log level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL '''
    logger.setLevel(level)


def log(level, msg, *args, **kwargs):
    ''' Print 'msg % args' with the severity 'level'.

    :arg level: the log level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    :arg msg: the message '''

    logger.log(level, msg, *args, **kwargs)


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
