# Some generic python utilities not really specific to our work.
import collections.abc
import warnings

from decorator import decorator
from petsc4py import PETSc

from pyop2.utils import cached_property  # noqa: F401
from pyop2.datatypes import ScalarType, as_cstr
from pyop2.datatypes import RealType     # noqa: F401
from pyop2.datatypes import IntType      # noqa: F401
from pyop2.datatypes import as_ctypes    # noqa: F401
from pyop2.mpi import MPI
from pyop3.utils import (  # noqa: F401
    OrderedSet,
    readonly,
    pairwise,
    steps,
    just_one,
    single_valued,
    is_single_valued,
    has_unique_entries,
    strictly_all,
    debug_assert,
    freeze,
    strict_int,
    StrictlyUniqueDict,
    invert,
    split_by,
    as_tuple,
)

import petsctools



# MPI key value for storing a per communicator universal identifier
FIREDRAKE_UID = MPI.Comm.Create_keyval()

RealType_c = as_cstr(RealType)
ScalarType_c = as_cstr(ScalarType)
IntType_c = as_cstr(IntType)

complex_mode = (petsctools.get_petscvariables()["PETSC_SCALAR"].lower() == "complex")

# Remove this (and update test suite) when Slate supports complex mode.
SLATE_SUPPORTS_COMPLEX = False


def _new_uid(comm):
    uid = comm.Get_attr(FIREDRAKE_UID)
    if uid is None:
        uid = 0
    comm.Set_attr(FIREDRAKE_UID, uid + 1)
    return uid


def _init():
    """Cause :func:`pyop2.init` to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    :func:`pyop2.init` if she wants to set a non-default option, for example
    to switch the debug or log level."""
    from pyop2 import op2
    from firedrake.parameters import parameters
    if not op2.initialised():
        op2.init(**parameters["pyop2_options"])


def unique(iterable):
    """ Return tuple of unique items in iterable, items must be hashable
    """
    # Use dict to preserve order and compare by hash
    unique_dict = {}
    for item in iterable:
        unique_dict[item] = None
    return tuple(unique_dict.keys())


def unique_name(name, nameset):
    """Return name if name is not in nameset, or a deterministic
    uniquified name if name is in nameset. The new name is inserted into
    nameset to prevent further name clashes."""

    if name not in nameset:
        nameset.add(name)
        return name

    idx = 0
    while True:
        newname = "%s_%d" % (name, idx)
        if newname in nameset:
            idx += 1
        else:
            nameset.add(name)
            return newname


def known_pyop2_safe(f):
    """Decorator to mark a function as being PyOP2 type-safe.

    This switches the current PyOP2 type checking mode to the value
    given by the parameter "type_check_safe_par_loops", and restores
    it after the function completes."""
    from firedrake.parameters import parameters

    def wrapper(f, *args, **kwargs):
        opts = parameters["pyop2_options"]
        check = opts["type_check"]
        safe = parameters["type_check_safe_par_loops"]
        if check == safe:
            return f(*args, **kwargs)
        opts["type_check"] = safe
        try:
            return f(*args, **kwargs)
        finally:
            opts["type_check"] = check
    return decorator(wrapper, f)


def tuplify(item):
    """Convert an object into a hashable equivalent.

    This is particularly useful for caching dictionaries of parameters such
    as `form_compiler_parameters` from :func:`firedrake.assemble.assemble`.

    :arg item: The object to attempt to 'tuplify'.
    :returns: The object interpreted as a tuple. For hashable objects this is
        simply a 1-tuple containing `item`. For dictionaries the function is
        called recursively on the values of the dict. For example,
        `{"a": 5, "b": 8}` returns `(("a", (5,)), ("b", (8,)))`.
    """
    if isinstance(item, collections.abc.Hashable):
        return (item,)

    if not isinstance(item, dict):
        raise ValueError(f"tuplify does not know how to handle objects of type {type(item)}")
    return tuple((k, tuplify(item[k])) for k in sorted(item))


def assert_empty(iterator):
    """Check that an iterator has been fully consumed.

    Raises
    ------
    AssertionError
        If the provided iterator is not empty.

    Notes
    -----
    This function should only be used for assertions (where the program is
    immediately terminated on failure). If the iterator is not empty then the
    latest value is discarded.

    """
    try:
        next(iterator)
        raise AssertionError("Iterator is not empty")
    except StopIteration:
        pass


# NOTE: Python 3.13 has warnings.deprecated which does exactly this
def deprecated(prefer=None, internal=False):
    """Decorator that emits a warning saying that the function is deprecated."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            msg = f"{fn.__qualname__} is deprecated and will be removed"
            if prefer:
                msg += f", please use {prefer} instead"
            warning_type = DeprecationWarning if internal else FutureWarning
            warnings.warn(msg, warning_type)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def safe_is(is_: PETSc.IS, *, comm=MPI.COMM_SELF) -> PETSc.IS:
    """Return a non-null index set.

    This function is useful because sometimes petsc4py returns index sets that
    are not correctly initialised.

    """
    return is_ if is_ else PETSc.IS().createStride(0, comm=comm).toGeneral()
