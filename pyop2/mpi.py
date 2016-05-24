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

from __future__ import absolute_import
from petsc4py import PETSc
from mpi4py import MPI  # noqa
from .utils import trim


COMM_WORLD = PETSc.COMM_WORLD.tompi4py()

COMM_SELF = PETSc.COMM_SELF.tompi4py()


def dup_comm(comm):
    """Duplicate a communicator for internal use.

    :arg comm: An mpi4py or petsc4py Comm object.

    :returns: A tuple of `(mpi4py.Comm, petsc4py.Comm)`.

    .. warning::

       This uses ``PetscCommDuplicate`` to create an internal
       communicator.  The petsc4py Comm thus returned will be
       collected (and ``MPI_Comm_free``d) when it goes out of scope.
       But the mpi4py comm is just a pointer at the underlying MPI
       handle.  So you need to hold on to both return values to ensure
       things work.  The collection of the petsc4py instance ensures
       the handles are all cleaned up."""
    if comm is None:
        comm = COMM_WORLD
    if isinstance(comm, MPI.Comm):
        comm = PETSc.Comm(comm)
    elif not isinstance(comm, PETSc.Comm):
        raise TypeError("Can't dup a %r" % type(comm))

    dcomm = comm.duplicate()
    comm = dcomm.tompi4py()
    return comm, dcomm


def collective(fn):
    extra = trim("""
    This function is logically collective over MPI ranks, it is an
    error to call it on fewer than all the ranks in MPI communicator.
    """)
    fn.__doc__ = "%s\n\n%s" % (trim(fn.__doc__), extra) if fn.__doc__ else extra
    return fn


# Install an exception hook to MPI Abort if an exception isn't caught
# see: https://groups.google.com/d/msg/mpi4py/me2TFzHmmsQ/sSF99LE0t9QJ
if COMM_WORLD.size > 1:
    import sys
    except_hook = sys.excepthook

    def mpi_excepthook(typ, value, traceback):
        except_hook(typ, value, traceback)
        COMM_WORLD.Abort(1)
    sys.excepthook = mpi_excepthook

import logging
logger = logging.getLogger("pyop2")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(('[%d] ' % COMM_WORLD.rank) +
                                       '%(name)s:%(levelname)s %(message)s'))
logger.addHandler(handler)
