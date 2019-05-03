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


from petsc4py import PETSc
from mpi4py import MPI  # noqa
import atexit
from pyop2.utils import trim


__all__ = ("COMM_WORLD", "COMM_SELF", "MPI", "dup_comm")

# These are user-level communicators, we never send any messages on
# them inside PyOP2.
COMM_WORLD = PETSc.COMM_WORLD.tompi4py()

COMM_SELF = PETSc.COMM_SELF.tompi4py()

# Exposition:
#
# To avoid PyOP2 library messages interfering with messages that the
# user might send on communicators, we duplicate any communicator
# passed in to PyOP2 and send our messages on this internal
# communicator.  This is equivalent to the way PETSc does things.
#
# To avoid unnecessarily duplicating communicators that we've already
# seen, we store information on both the inner and the outer
# communicator using MPI attributes, including a refcount.
#
# The references are as follows:
#
#     .-----------.       .------------.
#     |           |--->---|            |       .----------.
#     | User-Comm |       | PyOP2-Comm |--->---| Refcount |
#     |           |---<---|            |       '----------'
#     '-----------'       '------------'
#
# When we're asked to duplicate a communicator, we first check if it
# has a refcount (therefore it's a PyOP2 comm).  In which case we
# increment the refcount and return it.
#
# If it's not a PyOP2 comm, we check if it has an embedded PyOP2 comm,
# pull that out, increment the refcount and return it.
#
# If we've never seen this communicator before, we MPI_Comm_dup it,
# and set up the references with an initial refcount of 1.
#
# This is all handled in dup_comm.
#
# The matching free_comm is used to decrement the refcount on a
# duplicated communicator, eventually calling MPI_Comm_free when that
# refcount hits 0.  This is necessary since a design decision in
# mpi4py means that the user is responsible for calling MPI_Comm_free
# on any dupped communicators (rather than relying on the garbage collector).
#
# Finally, since it's difficult to know when all these communicators
# go out of scope, we register an atexit handler to clean up any
# outstanding duplicated communicators.


def delcomm_outer(comm, keyval, icomm):
    """Deleter for internal communicator, removes reference to outer comm.

    :arg comm: Outer communicator.
    :arg keyval: The MPI keyval, should be ``innercomm_keyval``.
    :arg icomm: The inner communicator, should have a reference to
        ``comm`.
    """
    if keyval != innercomm_keyval:
        raise ValueError("Unexpected keyval")
    ocomm = icomm.Get_attr(outercomm_keyval)
    if ocomm is None:
        raise ValueError("Inner comm does not have expected reference to outer comm")

    if ocomm != comm:
        raise ValueError("Inner comm has reference to non-matching outer comm")
    icomm.Delete_attr(outercomm_keyval)


# Refcount attribute for internal communicators
refcount_keyval = MPI.Comm.Create_keyval()

# Inner communicator attribute (attaches inner comm to user communicator)
innercomm_keyval = MPI.Comm.Create_keyval(delete_fn=delcomm_outer)

# Outer communicator attribute (attaches user comm to inner communicator)
outercomm_keyval = MPI.Comm.Create_keyval()

# List of internal communicators, must be freed at exit.
dupped_comms = []


def dup_comm(comm_in=None):
    """Given a communicator return a communicator for internal use.

    :arg comm_in: Communicator to duplicate.  If not provided,
        defaults to COMM_WORLD.

    :returns: An mpi4py communicator."""
    if comm_in is None:
        comm_in = COMM_WORLD
    if isinstance(comm_in, PETSc.Comm):
        comm_in = comm_in.tompi4py()
    elif not isinstance(comm_in, MPI.Comm):
        raise ValueError("Don't know how to dup a %r" % type(comm_in))
    if comm_in == MPI.COMM_NULL:
        return comm_in
    refcount = comm_in.Get_attr(refcount_keyval)
    if refcount is not None:
        # Passed an existing PyOP2 comm, return it
        comm_out = comm_in
        refcount[0] += 1
    else:
        # Check if communicator has an embedded PyOP2 comm.
        comm_out = comm_in.Get_attr(innercomm_keyval)
        if comm_out is None:
            # Haven't seen this comm before, duplicate it.
            comm_out = comm_in.Dup()
            comm_in.Set_attr(innercomm_keyval, comm_out)
            comm_out.Set_attr(outercomm_keyval, comm_in)
            # Refcount
            comm_out.Set_attr(refcount_keyval, [1])
            # Remember we need to destroy it.
            dupped_comms.append(comm_out)
        else:
            refcount = comm_out.Get_attr(refcount_keyval)
            if refcount is None:
                raise ValueError("Inner comm without a refcount")
            refcount[0] += 1
    return comm_out


# Comm used for compilation, stashed on the internal communicator
compilationcomm_keyval = MPI.Comm.Create_keyval()


def get_compilation_comm(comm):
    return comm.Get_attr(compilationcomm_keyval)


def set_compilation_comm(comm, inner):
    comm.Set_attr(compilationcomm_keyval, inner)


def free_comm(comm, remove=True):
    """Free an internal communicator.

    :arg comm: The communicator to free.
    :kwarg remove: Remove from list of dupped comms?

    This only actually calls MPI_Comm_free once the refcount drops to
    zero.
    """
    if comm == MPI.COMM_NULL:
        return
    refcount = comm.Get_attr(refcount_keyval)
    if refcount is None:
        # Not a PyOP2 communicator, check for an embedded comm.
        comm = comm.Get_attr(innercomm_keyval)
        if comm is None:
            raise ValueError("Trying to destroy communicator not known to PyOP2")
        refcount = comm.Get_attr(refcount_keyval)
        if refcount is None:
            raise ValueError("Inner comm without a refcount")

    refcount[0] -= 1

    if refcount[0] == 0:
        ocomm = comm.Get_attr(outercomm_keyval)
        if ocomm is not None:
            icomm = ocomm.Get_attr(innercomm_keyval)
            if icomm is None:
                raise ValueError("Outer comm does not reference inner comm ")
            else:
                ocomm.Delete_attr(innercomm_keyval)
            del icomm
        if remove:
            # Only do this if not called from free_comms.
            dupped_comms.remove(comm)
        compilation_comm = get_compilation_comm(comm)
        if compilation_comm is not None:
            compilation_comm.Free()
        comm.Free()


@atexit.register
def free_comms():
    """Free all outstanding communicators."""
    while dupped_comms:
        c = dupped_comms.pop()
        refcount = c.Get_attr(refcount_keyval)
        for _ in range(refcount[0]):
            free_comm(c, remove=False)
    for kv in [refcount_keyval,
               innercomm_keyval,
               outercomm_keyval,
               compilationcomm_keyval]:
        MPI.Comm.Free_keyval(kv)


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
        sys.stderr.flush()
        COMM_WORLD.Abort(1)
    sys.excepthook = mpi_excepthook
