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
import os
from pyop2.configuration import configuration
from pyop2.exceptions import CompilationError
from pyop2.logger import warning, debug
from pyop2.utils import trim


__all__ = ("COMM_WORLD", "COMM_SELF", "MPI", "internal_comm", "is_pyop2_comm", "incref", "decref", "PyOP2Comm")

# These are user-level communicators, we never send any messages on
# them inside PyOP2.
COMM_WORLD = PETSc.COMM_WORLD.tompi4py()
COMM_WORLD.Set_name("PYOP2_COMM_WORLD")

COMM_SELF = PETSc.COMM_SELF.tompi4py()
COMM_SELF.Set_name("PYOP2_COMM_SELF")

PYOP2_FINALIZED = False

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


def collective(fn):
    extra = trim("""
    This function is logically collective over MPI ranks, it is an
    error to call it on fewer than all the ranks in MPI communicator.
    """)
    fn.__doc__ = "%s\n\n%s" % (trim(fn.__doc__), extra) if fn.__doc__ else extra
    return fn


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

# Comm used for compilation, stashed on the internal communicator
compilationcomm_keyval = MPI.Comm.Create_keyval()

# List of internal communicators, must be freed at exit.
dupped_comms = []


def is_pyop2_comm(comm):
    """Returns `True` if `comm` is a PyOP2 communicator,
    False if `comm` another communicator.
    Raises exception if `comm` is not a communicator.

    :arg comm: Communicator to query
    """
    global PYOP2_FINALIZED
    if isinstance(comm, PETSc.Comm):
        ispyop2comm = False
    elif comm == MPI.COMM_NULL:
        if PYOP2_FINALIZED is False:
            # ~ import pytest; pytest.set_trace()
            raise ValueError("COMM_NULL")
            ispyop2comm = True
        else:
            ispyop2comm = True
    elif isinstance(comm, MPI.Comm):
        ispyop2comm = bool(comm.Get_attr(refcount_keyval))
    else:
        raise ValueError(f"Argument passed to is_pyop2_comm() is a {type(comm)}, which is not a recognised comm type")
    return ispyop2comm


def pyop2_comm_status():
    """ Prints the reference counts for all comms PyOP2 has duplicated
    """
    print('PYOP2 Communicator reference counts:')
    print('| Communicator name                      | Count |')
    print('==================================================')
    for comm in dupped_comms:
        if comm == MPI.COMM_NULL:
            null = 'COMM_NULL'
            print(f'| {null:39}| {0:5d} |')
        else:
            refcount = comm.Get_attr(refcount_keyval)[0]
            if refcount is None:
                refcount = -999
            print(f'| {comm.name:39}| {refcount:5d} |')


class PyOP2Comm:
    """ Suitable for using a PyOP2 internal communicator suitably
    incrementing and decrementing the comm.
    """
    def __init__(self, comm):
        self.comm = comm
        self._comm = None

    def __enter__(self):
        self._comm = internal_comm(self.comm)
        return self._comm

    def __exit__(self, exc_type, exc_value, traceback):
        decref(self._comm)
        self._comm = None


def internal_comm(comm):
    """ Creates an internal comm from the comm passed in
    This happens on nearly every PyOP2 object so this avoids unnecessary
    repetition.
    :arg comm: A communicator or None

    :returns pyop2_comm: A PyOP2 internal communicator
    """
    if comm is None:
        # None will be the default when creating most objects
        pyop2_comm = dup_comm(COMM_WORLD)
    elif is_pyop2_comm(comm):
        # Increase the reference count and return same comm if
        # already an internal communicator
        incref(comm)
        pyop2_comm = comm
    elif isinstance(comm, PETSc.Comm):
        # Convert PETSc.Comm to mpi4py.MPI.Comm
        pyop2_comm = dup_comm(comm.tompi4py())
    elif comm == MPI.COMM_NULL:
        # Ensure comm is not the NULL communicator
        raise ValueError("MPI_COMM_NULL passed to internal_comm()")
    elif not isinstance(comm, MPI.Comm):
        # If it is not an MPI.Comm raise error
        raise ValueError("Don't know how to dup a %r" % type(comm))
    else:
        pyop2_comm = dup_comm(comm)
    return pyop2_comm


def incref(comm):
    """ Increment communicator reference count
    """
    assert is_pyop2_comm(comm)
    refcount = comm.Get_attr(refcount_keyval)
    refcount[0] += 1


def decref(comm):
    """ Decrement communicator reference count
    """
    if not PYOP2_FINALIZED:
        assert is_pyop2_comm(comm)
        refcount = comm.Get_attr(refcount_keyval)
        refcount[0] -= 1
        if refcount[0] == 0:
            dupped_comms.remove(comm)
            free_comm(comm)
    elif comm == MPI.COMM_NULL:
        pass
    else:
        free_comm(comm)


def dup_comm(comm_in):
    """Given a communicator return a communicator for internal use.

    :arg comm_in: Communicator to duplicate

    :returns: An mpi4py communicator."""
    assert not is_pyop2_comm(comm_in)

    # Check if communicator has an embedded PyOP2 comm.
    comm_out = comm_in.Get_attr(innercomm_keyval)
    if comm_out is None:
        # Haven't seen this comm before, duplicate it.
        comm_out = comm_in.Dup()
        comm_in.Set_attr(innercomm_keyval, comm_out)
        comm_out.Set_attr(outercomm_keyval, comm_in)
        # Name
        # TODO: replace id() with .py2f() ???
        comm_out.Set_name(f"{comm_in.name or id(comm_in)}_DUP")
        # Refcount
        comm_out.Set_attr(refcount_keyval, [0])
        incref(comm_out)
        # Remember we need to destroy it.
        dupped_comms.append(comm_out)
    elif is_pyop2_comm(comm_out):
        # Inner comm is a PyOP2 comm, return it
        incref(comm_out)
    else:
        raise ValueError("Inner comm is not a PyOP2 comm")
    return comm_out


@collective
def create_split_comm(comm):
    if MPI.VERSION >= 3:
        debug("Creating compilation communicator using MPI_Split_type")
        split_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        debug("Finished creating compilation communicator using MPI_Split_type")
    else:
        debug("Creating compilation communicator using MPI_Split + filesystem")
        import tempfile
        if comm.rank == 0:
            if not os.path.exists(configuration["cache_dir"]):
                os.makedirs(configuration["cache_dir"], exist_ok=True)
            tmpname = tempfile.mkdtemp(prefix="rank-determination-",
                                       dir=configuration["cache_dir"])
        else:
            tmpname = None
        tmpname = comm.bcast(tmpname, root=0)
        if tmpname is None:
            raise CompilationError("Cannot determine sharedness of filesystem")
        # Touch file
        debug("Made tmpdir %s" % tmpname)
        with open(os.path.join(tmpname, str(comm.rank)), "wb"):
            pass
        comm.barrier()
        import glob
        ranks = sorted(int(os.path.basename(name))
                       for name in glob.glob("%s/[0-9]*" % tmpname))
        debug("Creating compilation communicator using filesystem colors")
        split_comm = comm.Split(color=min(ranks), key=comm.rank)
        debug("Finished creating compilation communicator using filesystem colors")
    # Name
    split_comm.Set_name(f"{comm.name or id(comm)}_COMPILATION")
    # Refcount
    split_comm.Set_attr(refcount_keyval, [0])
    incref(split_comm)
    return split_comm


def get_compilation_comm(comm):
    return comm.Get_attr(compilationcomm_keyval)


def set_compilation_comm(comm, inner):
    """Set the compilation communicator.

    :arg comm: A PyOP2 Communicator
    :arg inner: The compilation communicator
    """
    # Ensure `comm` is a PyOP2 comm
    if not is_pyop2_comm(comm):
        raise ValueError("Compilation communicator must be stashed on a PyOP2 comm")

    # Check if the compilation communicator is already set
    old_inner = comm.Get_attr(compilationcomm_keyval)
    if old_inner is not None:
        if is_pyop2_comm(old_inner):
            raise ValueError("Compilation communicator is not a PyOP2 comm, something is very broken!")
        else:
            decref(old_inner)

    if not is_pyop2_comm(inner):
        raise ValueError(
            "Communicator used for compilation communicator must be a PyOP2 communicator.\n"
            "Use pyop2.mpi.dup_comm() to create a PyOP2 comm from an existing comm.")
    else:
        # Stash `inner` as an attribute on `comm`
        comm.Set_attr(compilationcomm_keyval, inner)


@collective
def compilation_comm(comm):
    """Get a communicator for compilation.

    :arg comm: The input communicator, must be a PyOP2 comm.
    :returns: A communicator used for compilation (may be smaller)
    """
    if not is_pyop2_comm(comm):
        raise ValueError("Compilation communicator is not a PyOP2 comm")
    # Should we try and do node-local compilation?
    if configuration["node_local_compilation"]:
        retcomm = get_compilation_comm(comm)
        if retcomm is not None:
            debug("Found existing compilation communicator")
            debug(f"{retcomm.name}")
        else:
            retcomm = create_split_comm(comm)
            set_compilation_comm(comm, retcomm)
            # Add to list of known duplicated comms
            debug(f"Appending compiler comm {retcomm.name} to list of comms")
            dupped_comms.append(retcomm)
    else:
        retcomm = comm
    incref(retcomm)
    return retcomm


def free_comm(comm):
    """Free an internal communicator.

    :arg comm: The communicator to free.
    :kwarg remove: Remove from list of dupped comms?

    This only actually calls MPI_Comm_free once the refcount drops to
    zero.
    """
    # ~ if isinstance(comm, list):
    # ~ import pytest; pytest.set_trace()
    if comm != MPI.COMM_NULL:
        assert is_pyop2_comm(comm)
        ocomm = comm.Get_attr(outercomm_keyval)
        if isinstance(ocomm, list):
            # No idea why this happens!?
            raise ValueError("Why have we got a list!?")
        if ocomm is not None:
            icomm = ocomm.Get_attr(innercomm_keyval)
            if icomm is None:
                raise ValueError("Outer comm does not reference inner comm ")
            else:
                ocomm.Delete_attr(innercomm_keyval)
            del icomm
        try:
            dupped_comms.remove(comm)
        except ValueError:
            debug(f"{comm.name} is not in list of known comms, probably already freed")
            debug(f"Known comms are {[d.name for d in dupped_comms if d != MPI.COMM_NULL]}")
        compilation_comm = get_compilation_comm(comm)
        if compilation_comm == MPI.COMM_NULL:
            comm.Delete_attr(compilationcomm_keyval)
        elif compilation_comm is not None:
            free_comm(compilation_comm)
            comm.Delete_attr(compilationcomm_keyval)
        comm.Free()
    else:
        warning('Attempt to free MPI_COMM_NULL')


@atexit.register
def free_comms():
    """Free all outstanding communicators."""
    # Collect garbage as it may hold on to communicator references
    global PYOP2_FINALIZED
    PYOP2_FINALIZED = True
    debug("PyOP2 Finalizing")
    debug("Calling gc.collect()")
    import gc
    gc.collect()
    pyop2_comm_status()
    print(dupped_comms)
    debug(f"Freeing comms in list (length {len(dupped_comms)})")
    while dupped_comms:
        c = dupped_comms[-1]
        if is_pyop2_comm(c):
            refcount = c.Get_attr(refcount_keyval)
            debug(f"Freeing {c.name}, which has refcount {refcount[0]}")
        else:
            debug("Freeing non PyOP2 comm in `free_comms()`")
        free_comm(c)
    for kv in [refcount_keyval,
               innercomm_keyval,
               outercomm_keyval,
               compilationcomm_keyval]:
        MPI.Comm.Free_keyval(kv)


def hash_comm(comm):
    """Return a hashable identifier for a communicator."""
    assert is_pyop2_comm(comm)
    return id(comm)


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
