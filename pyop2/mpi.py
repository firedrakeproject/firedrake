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

"""PyOP2 MPI communicator."""

from decorator import decorator
from mpi4py import MPI as _MPI
from utils import trim


def collective(fn):
    extra = trim("""
    This function is logically collective over MPI ranks, it is an
    error to call it on fewer than all the ranks in MPI communicator.
    """)
    fn.__doc__ = "%s\n\n%s" % (trim(fn.__doc__), extra) if fn.__doc__ else extra
    return fn


def _check_comm(comm):
    if isinstance(comm, int):
        # If it's come from Fluidity where an MPI_Comm is just an integer.
        return _MPI.Comm.f2py(comm)
    try:
        return comm if isinstance(comm, _MPI.Comm) else comm.tompi4py()
    except AttributeError:
        raise TypeError("MPI communicator must be of type mpi4py.MPI.Comm")


class MPIConfig(object):

    def __init__(self):
        self.COMM = _MPI.COMM_WORLD

    @property
    def parallel(self):
        """Are we running in parallel?"""
        return self.comm.size > 1

    @property
    def comm(self):
        """The MPI Communicator used by PyOP2."""
        return self.COMM

    @comm.setter
    @collective
    def comm(self, comm):
        """Set the MPI communicator for parallel communication.

        .. note:: The communicator must be of type :py:class:`mpi4py.MPI.Comm`
        or implement a method :py:meth:`tompi4py` to be converted to one."""
        self.COMM = _check_comm(comm)

    def rank_zero(self, f):
        """Decorator for executing a function only on MPI rank zero."""
        def wrapper(f, *args, **kwargs):
            if self.comm.rank == 0:
                return f(*args, **kwargs)
        return decorator(wrapper, f)

MPI = MPIConfig()

# Install an exception hook to MPI Abort if an exception isn't caught
# see: https://groups.google.com/d/msg/mpi4py/me2TFzHmmsQ/sSF99LE0t9QJ
if MPI.parallel:
    import sys
    except_hook = sys.excepthook

    def mpi_excepthook(typ, value, traceback):
        except_hook(typ, value, traceback)
        MPI.comm.Abort(1)
    sys.excepthook = mpi_excepthook
