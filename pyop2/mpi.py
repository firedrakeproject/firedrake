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
from itertools import count
from functools import wraps
import atexit
import gc
import glob
import os
import tempfile
import weakref

from pyop2.configuration import configuration
from pyop2.exceptions import CompilationError
from pyop2.logger import debug, logger, DEBUG
from pyop2.utils import trim


__all__ = (
    "COMM_WORLD",
    "COMM_SELF",
    "MPI",
    "internal_comm",
    "is_pyop2_comm",
    "incref",
    "decref",
    "temp_internal_comm"
)

# These are user-level communicators, we never send any messages on
# them inside PyOP2.
COMM_WORLD = PETSc.COMM_WORLD.tompi4py().Dup()
COMM_WORLD.Set_name("PYOP2_COMM_WORLD")

COMM_SELF = PETSc.COMM_SELF.tompi4py().Dup()
COMM_SELF.Set_name("PYOP2_COMM_SELF")

# Creation index counter
_COMM_CIDX = count()
# Dict of internal communicators, keyed by creation index, to be freed at exit.
_DUPED_COMM_DICT = {}
# Flag to indicate whether we are in cleanup (at exit)
PYOP2_FINALIZED = False
# Flag for outputting information at the end of testing (do not abuse!)
_running_on_ci = bool(os.environ.get('PYOP2_CI_TESTS'))


class PyOP2CommError(ValueError):
    pass

# ============
# Exposition:
# ============
#
# To avoid PyOP2 library messages interfering with messages that the
# user might send on communicators, we duplicate any communicator
# passed in to PyOP2 and send our messages on this internal
# communicator.  This is equivalent to the way PETSc does things.
#
# To avoid unnecessarily duplicating communicators that we've already
# seen, we store information on both the inner and the outer
# communicator using MPI attributes. In addition we store the reference
# count and creation index as attributes on PyOP2 comms.
#
# The references are as follows:
#
#    User Facing Comms       PyOP2 Comms           DUPED
#      .-----------.       .-------------.         COMM
#      | User-Comm |------>| PyOP2-Comm  |         DICT
#      |```````````|       |`````````````|       .-------.
#      |           |<------| refcount    |<------| cidx  |
#      |           |       | cidx        |       |```````|
#      '-----------'       '-------------'       |       |
#                              |    ^            |       |
#                              |    |            |       |
#                              v    |            |       |
#                          .-------------.       |       |
#                          | Compilation |       |       |
#                          | Comm        |       |.......|
#                          |`````````````|<------| cidx  |
#                          | refcount    |       '-------'
#                          | cidx        |
#                          '-------------'
#
# Creation:
# ----------
# When we're asked to for an internal communicator, we first check if it
# has a refcount (therefore it's a PyOP2 comm). In which case we
# increment the refcount and return it.
#
# If it's not a PyOP2 comm, we check if it has an embedded PyOP2 comm,
# pull that out, increment the refcount and return it.
#
# If we've never seen this communicator before, we MPI_Comm_dup it,
# and set up the references with an initial refcount of 2:
#   - One for the returned PyOP2 comm
#   - One for the reference held by the internal dictionary of created
#     comms
# We also assign the comm a creation index (cidx).
#
# Something similar happens for compilation communicators.
#
# This is all handled by the user-facing functions internal_comm() and
# compilation_comm().
#
# Destruction:
# -------------
# Freeing communicators is tricky as the Python cyclic garbage
# collector can cause decref to be called. Unless the garage collector
# is called simultaneously on all ranks (unlikely to happen) the
# reference count for an internal comm will not agree across all ranks.
# To avoid the situation where Free() is called on some ranks but not
# others we maintain one reference to any duplicated comm in the global
# _DUPED_COMM_DICT.
#
# The user is responsible for calling MPI_Comm_free on any user
# communicators. When a user destroys a the MPI callback delcomm_outer()
# ensures that the corresponding PyOP2 comms are properly freed.
#
# Cleanup:
# ---------
# Finally, we register an atexit handler _free_comms() to clean up any
# outstanding duplicated communicators by freeing any remaining entries
# in _DUPED_COMM_DICT. Since the interpreter is shutting down, it is
# necessary to skip some checks, this is done by setting the
# PYOP2_FINALISED flag.


if configuration["spmd_strict"]:
    def collective(fn):
        extra = trim("""
        This function is logically collective over MPI ranks, it is an
        error to call it on fewer than all the ranks in MPI communicator.
        PYOP2_SPMD_STRICT=1 is in your environment and function calls will be
        guarded by a barrier where possible.
        """)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            comms = filter(
                lambda arg: isinstance(arg, MPI.Comm),
                args + tuple(kwargs.values())
            )
            try:
                comm = next(comms)
            except StopIteration:
                if args and hasattr(args[0], "comm"):
                    comm = args[0].comm
                else:
                    comm = None

            if comm is None:
                debug(
                    "`@collective` wrapper found no communicators in args or kwargs, "
                    "this means that the call is implicitly collective over an "
                    "unknown communicator. "
                    f"The following call to {fn.__module__}.{fn.__qualname__} is "
                    "not protected by an MPI barrier."
                )
                subcomm = ", UNKNOWN Comm"
            else:
                subcomm = f", {comm.name} R{comm.rank}"

            debug_string_pt1 = f"{COMM_WORLD.name} R{COMM_WORLD.rank}{subcomm}: "
            debug_string_pt2 = f" {fn.__module__}.{fn.__qualname__}"
            debug(debug_string_pt1 + "Entering" + debug_string_pt2)
            if comm is not None:
                comm.Barrier()
            value = fn(*args, **kwargs)
            debug(debug_string_pt1 + "Leaving" + debug_string_pt2)
            if comm is not None:
                comm.Barrier()
            return value

        wrapper.__doc__ = f"{trim(fn.__doc__)}\n\n{extra}" if fn.__doc__ else extra
        return wrapper
else:
    def collective(fn):
        extra = trim("""
        This function is logically collective over MPI ranks, it is an
        error to call it on fewer than all the ranks in MPI communicator.
        You can set PYOP2_SPMD_STRICT=1 in your environment to try and catch
        non-collective calls.
        """)
        fn.__doc__ = f"{trim(fn.__doc__)}\n\n{extra}" if fn.__doc__ else extra
        return fn


def delcomm_outer(comm, keyval, icomm):
    """Deleter for internal communicator, removes reference to outer comm.
    Generalised to also delete compilation communicators.

    :arg comm: Outer communicator.
    :arg keyval: The MPI keyval, should be ``innercomm_keyval``.
    :arg icomm: The inner communicator, should have a reference to
        ``comm``.
    """
    # Use debug printer that is safe to use at exit time
    debug = finalize_safe_debug()
    if keyval not in (innercomm_keyval, compilationcomm_keyval):
        raise PyOP2CommError("Unexpected keyval")

    if keyval == innercomm_keyval:
        debug(f'Deleting innercomm keyval on {comm.name}')
    if keyval == compilationcomm_keyval:
        debug(f'Deleting compilationcomm keyval on {comm.name}')

    ocomm = icomm.Get_attr(outercomm_keyval)
    if ocomm is None:
        raise PyOP2CommError("Inner comm does not have expected reference to outer comm")

    if ocomm != comm:
        raise PyOP2CommError("Inner comm has reference to non-matching outer comm")
    icomm.Delete_attr(outercomm_keyval)

    # An inner comm may or may not hold a reference to a compilation comm
    comp_comm = icomm.Get_attr(compilationcomm_keyval)
    if comp_comm is not None:
        debug('Removing compilation comm on inner comm')
        decref(comp_comm)
        icomm.Delete_attr(compilationcomm_keyval)

    # Once we have removed the reference to the inner/compilation comm we can free it
    cidx = icomm.Get_attr(cidx_keyval)
    cidx = cidx[0]
    del _DUPED_COMM_DICT[cidx]
    gc.collect()
    refcount = icomm.Get_attr(refcount_keyval)
    if refcount[0] > 1:
        # In the case where `comm` is a custom user communicator there may be references
        # to the inner comm still held and this is not an issue, but there is not an
        # easy way to distinguish this case, so we just log the event.
        debug(
            f"There are still {refcount[0]} references to {comm.name}, "
            "this will cause deadlock if the communicator has been incorrectly freed"
        )
    icomm.Free()


# Reference count, creation index, inner/outer/compilation communicator
# attributes for internal communicators
refcount_keyval = MPI.Comm.Create_keyval()
cidx_keyval = MPI.Comm.Create_keyval()
innercomm_keyval = MPI.Comm.Create_keyval(delete_fn=delcomm_outer)
outercomm_keyval = MPI.Comm.Create_keyval()
compilationcomm_keyval = MPI.Comm.Create_keyval(delete_fn=delcomm_outer)
comm_cache_keyval = MPI.Comm.Create_keyval()


def is_pyop2_comm(comm):
    """Returns ``True`` if ``comm`` is a PyOP2 communicator,
    False if `comm` another communicator.
    Raises exception if ``comm`` is not a communicator.

    :arg comm: Communicator to query
    """
    if isinstance(comm, PETSc.Comm):
        ispyop2comm = False
    elif comm == MPI.COMM_NULL:
        raise PyOP2CommError("Communicator passed to is_pyop2_comm() is COMM_NULL")
    elif isinstance(comm, MPI.Comm):
        ispyop2comm = bool(comm.Get_attr(refcount_keyval))
    else:
        raise PyOP2CommError(f"Argument passed to is_pyop2_comm() is a {type(comm)}, which is not a recognised comm type")
    return ispyop2comm


def pyop2_comm_status():
    """ Return string containing a table of the reference counts for all
    communicators  PyOP2 has duplicated.
    """
    status_string = 'PYOP2 Communicator reference counts:\n'
    status_string += '| Communicator name                      | Count |\n'
    status_string += '==================================================\n'
    for comm in _DUPED_COMM_DICT.values():
        if comm == MPI.COMM_NULL:
            null = 'COMM_NULL'
            status_string += f'| {null:39}| {0:5d} |\n'
        else:
            refcount = comm.Get_attr(refcount_keyval)[0]
            if refcount is None:
                refcount = -999
            status_string += f'| {comm.name:39}| {refcount:5d} |\n'
    return status_string


class temp_internal_comm:
    """ Use a PyOP2 internal communicator and
    increment and decrement the internal comm.
    :arg comm: Any communicator
    """
    def __init__(self, comm):
        self.user_comm = comm
        self.internal_comm = internal_comm(self.user_comm, self)

    def __enter__(self):
        """ Returns an internal comm that will be safely decref'd
        when the context manager is destroyed

        :returns pyop2_comm: A PyOP2 internal communicator
        """
        return self.internal_comm

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def internal_comm(comm, obj):
    """ Creates an internal comm from the user comm.
    If comm is None, create an internal communicator from COMM_WORLD
    :arg comm: A communicator or None
    :arg obj: The object which the comm is an attribute of
    (usually `self`)

    :returns pyop2_comm: A PyOP2 internal communicator
    """
    # Parse inputs
    if comm is None:
        # None will be the default when creating most objects
        comm = COMM_WORLD
    elif isinstance(comm, PETSc.Comm):
        comm = comm.tompi4py()

    # Check for invalid inputs
    if comm == MPI.COMM_NULL:
        raise PyOP2CommError("MPI_COMM_NULL passed to internal_comm()")
    elif not isinstance(comm, MPI.Comm):
        raise PyOP2CommError("Don't know how to dup a %r" % type(comm))

    # Handle a valid input
    if is_pyop2_comm(comm):
        incref(comm)
        pyop2_comm = comm
    else:
        pyop2_comm = dup_comm(comm)
    weakref.finalize(obj, decref, pyop2_comm)
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
    if comm == MPI.COMM_NULL:
        # This case occurs if the the outer communicator has already been freed by
        # the user
        debug("Cannot decref an already freed communicator")
    else:
        assert is_pyop2_comm(comm)
        refcount = comm.Get_attr(refcount_keyval)
        refcount[0] -= 1
        # Freeing the internal comm is handled by the destruction of the user comm
        if refcount[0] < 1:
            raise PyOP2CommError("Reference count is less than 1, decref called too many times")


def dup_comm(comm_in):
    """Given a communicator return a communicator for internal use.

    :arg comm_in: Communicator to duplicate

    :returns internal_comm: An internal (PyOP2) communicator."""
    assert not is_pyop2_comm(comm_in)

    # Check if communicator has an embedded PyOP2 comm.
    internal_comm = comm_in.Get_attr(innercomm_keyval)
    if internal_comm is None:
        # Haven't seen this comm before, duplicate it.
        internal_comm = comm_in.Dup()
        comm_in.Set_attr(innercomm_keyval, internal_comm)
        internal_comm.Set_attr(outercomm_keyval, comm_in)
        # Name
        internal_comm.Set_name(f"{comm_in.name or comm_in.py2f()}_DUP")
        # Refcount
        internal_comm.Set_attr(refcount_keyval, [1])
        incref(internal_comm)
        # Remember we need to destroy it.
        debug(f"Appending comm {internal_comm.name} to list of known comms")
        cidx = next(_COMM_CIDX)
        internal_comm.Set_attr(cidx_keyval, [cidx])
        _DUPED_COMM_DICT[cidx] = internal_comm
    elif is_pyop2_comm(internal_comm):
        # Inner comm is a PyOP2 comm, return it
        incref(internal_comm)
    else:
        raise PyOP2CommError("Inner comm is not a PyOP2 comm")
    return internal_comm


@collective
def create_split_comm(comm):
    """ Create a split communicator based on either shared memory access
    if using MPI >= 3, or shared local disk access if using MPI <= 3.
    Used internally for creating compilation communicators

    :arg comm: A communicator to split

    :return split_comm: A split communicator
    """
    if MPI.VERSION >= 3:
        debug("Creating compilation communicator using MPI_Split_type")
        split_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        debug("Finished creating compilation communicator using MPI_Split_type")
    else:
        debug("Creating compilation communicator using MPI_Split + filesystem")
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
        ranks = sorted(int(os.path.basename(name))
                       for name in glob.glob("%s/[0-9]*" % tmpname))
        debug("Creating compilation communicator using filesystem colors")
        split_comm = comm.Split(color=min(ranks), key=comm.rank)
        debug("Finished creating compilation communicator using filesystem colors")
    # Name
    split_comm.Set_name(f"{comm.name or comm.py2f()}_COMPILATION")
    # Outer communicator
    split_comm.Set_attr(outercomm_keyval, comm)
    # Refcount
    split_comm.Set_attr(refcount_keyval, [1])
    incref(split_comm)
    return split_comm


def get_compilation_comm(comm):
    return comm.Get_attr(compilationcomm_keyval)


def set_compilation_comm(comm, comp_comm):
    """Stash the compilation communicator (``comp_comm``) on the
    PyOP2 communicator ``comm``

    :arg comm: A PyOP2 Communicator
    :arg comp_comm: The compilation communicator
    """
    if not is_pyop2_comm(comm):
        raise PyOP2CommError("Compilation communicator must be stashed on a PyOP2 comm")

    # Check if the compilation communicator is already set
    old_comp_comm = comm.Get_attr(compilationcomm_keyval)

    if not is_pyop2_comm(comp_comm):
        raise PyOP2CommError(
            "Communicator used for compilation communicator must be a PyOP2 communicator.\n"
            "Use pyop2.mpi.dup_comm() to create a PyOP2 comm from an existing comm.")
    else:
        if old_comp_comm is not None:
            # Clean up old_comp_comm before setting new one
            if not is_pyop2_comm(old_comp_comm):
                raise PyOP2CommError("Compilation communicator is not a PyOP2 comm, something is very broken!")
            gc.collect()
            decref(old_comp_comm)
        # Stash `comp_comm` as an attribute on `comm`
        comm.Set_attr(compilationcomm_keyval, comp_comm)
        # NB: Set_attr calls the delete method for the
        # compilationcomm_keyval freeing old_comp_comm


@collective
def compilation_comm(comm, obj):
    """Get a communicator for compilation.

    :arg comm: The input communicator, must be a PyOP2 comm.
    :arg obj: The object which the comm is an attribute of
    (usually `self`)

    :returns: A communicator used for compilation (may be smaller)
    """
    if not is_pyop2_comm(comm):
        raise PyOP2CommError("Communicator is not a PyOP2 comm")
    # Should we try and do node-local compilation?
    if configuration["node_local_compilation"]:
        comp_comm = get_compilation_comm(comm)
        if comp_comm is not None:
            debug("Found existing compilation communicator")
            debug(f"{comp_comm.name}")
        else:
            comp_comm = create_split_comm(comm)
            set_compilation_comm(comm, comp_comm)
            # Add to list of known duplicated comms
            debug(f"Appending compiler comm {comp_comm.name} to list of known comms")
            cidx = next(_COMM_CIDX)
            comp_comm.Set_attr(cidx_keyval, [cidx])
            _DUPED_COMM_DICT[cidx] = comp_comm
    else:
        comp_comm = comm
    incref(comp_comm)
    weakref.finalize(obj, decref, comp_comm)
    return comp_comm


def finalize_safe_debug():
    ''' Return function for debug output.

    When Python is finalizing the logging module may be finalized before we have
    finished writing debug information. In this case we fall back to using the
    Python `print` function to output debugging information.

    Furthermore, we always want to see this finalization information when
    running the CI tests.
    '''
    global debug
    if PYOP2_FINALIZED:
        if logger.level > DEBUG and not _running_on_ci:
            debug = lambda string: None
        else:
            debug = lambda string: print(string)
    return debug


@atexit.register
def _free_comms():
    """Free all outstanding communicators."""
    global PYOP2_FINALIZED
    PYOP2_FINALIZED = True
    debug = finalize_safe_debug()
    debug("PyOP2 Finalizing")
    # Collect garbage as it may hold on to communicator references

    debug("Calling gc.collect()")
    gc.collect()
    debug("STATE0")
    debug(pyop2_comm_status())

    debug("Freeing PYOP2_COMM_WORLD")
    COMM_WORLD.Free()
    debug("STATE1")
    debug(pyop2_comm_status())

    debug("Freeing PYOP2_COMM_SELF")
    COMM_SELF.Free()
    debug("STATE2")
    debug(pyop2_comm_status())
    debug(f"Freeing comms in list (length {len(_DUPED_COMM_DICT)})")
    for key in sorted(_DUPED_COMM_DICT.keys(), reverse=True):
        comm = _DUPED_COMM_DICT[key]
        if comm != MPI.COMM_NULL:
            refcount = comm.Get_attr(refcount_keyval)
            debug(f"Freeing {comm.name}, with index {key}, which has refcount {refcount[0]}")
            comm.Free()
        del _DUPED_COMM_DICT[key]
    for kv in [
        refcount_keyval,
        innercomm_keyval,
        outercomm_keyval,
        compilationcomm_keyval,
        comm_cache_keyval
    ]:
        MPI.Comm.Free_keyval(kv)


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
