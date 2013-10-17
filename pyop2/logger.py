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

import logging
from mpi import MPI

# Define colors
RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

logger = logging.getLogger('pyop2')
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter(('[%d] ' % MPI.comm.rank if MPI.parallel else '') +
                                   '%(name)s:%(levelname)s %(message)s'))
logger.addHandler(_ch)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


def set_log_level(level):
    '''Set the log level of the PyOP2 logger.

    :arg level: the log level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL '''
    logger.setLevel(level)


def info_red(message):
    ''' Write info message in red.

    :arg message: the message to be printed. '''
    info(RED % message)


def info_green(message):
    ''' Write info message in green.

    :arg message: the message to be printed. '''
    info(GREEN % message)


def info_blue(message):
    ''' Write info message in blue.

    :arg message: the message to be printed. '''
    info(BLUE % message)


def log(level, *args, **kwargs):
    ''' Print message at given debug level.

    :arg level: the log level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    :arg message: the message to be printed. '''

    logger.log(level, *args, **kwargs)
