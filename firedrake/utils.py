# Some generic python utilities not really specific to our work.
import collections.abc
import functools
import warnings
from typing import Callable, Self, Hashable

from decorator import decorator
from petsc4py import PETSc

from pyop3.collections import OrderedSet, StrictlyUniqueDict, StrictlyUniqueDefaultDict
from pyop3.dtypes import ScalarType, as_cstr
from pyop3.dtypes import RealType, IntType, as_ctypes     # noqa: F401
from pyop3.mpi import MPI
from pyop3.cache import cached_method
from pyop3.utils import (  # noqa: F401
    readonly,
    pairwise,
    steps,
    just_one,
    pretty_type,
    single_valued,
    is_single_valued,
    has_unique_entries,
    strictly_all,
    debug_assert,
    freeze,
    strict_int,
    invert,
    split_by,
    as_tuple,
    is_sorted,
    unique_name as op3_unique_name,
)

from functools import cache
from firedrake.exceptions import UnrecognisedDeviceError
import petsctools



# MPI key value for storing a per communicator universal identifier
FIREDRAKE_UID = MPI.Comm.Create_keyval()

RealType_c = as_cstr(RealType)
ScalarType_c = as_cstr(ScalarType)
IntType_c = as_cstr(IntType)

complex_mode = (petsctools.get_petscvariables()["PETSC_SCALAR"].lower() == "complex")

# Remove this (and update test suite) when Slate supports complex mode.
SLATE_SUPPORTS_COMPLEX = False


@cache
def get_device_type() -> str | None:
    r"""Get PETSc device type.

    Attempt to initialise a GPU and return the type of GPU
    identified by PETSc.

    Returns
    -------
    str | None
        The PETSc device type, or `None` if no device is found.

    """
    try:
        dev = PETSc.Device.create()
    except PETSc.Error:
        # Could not initialise device - not a failure condition as this could
        # be a GPU-enabled PETSc installation running on a CPU-only host.
        return None
    dev_type = dev.getDeviceType()
    dev.destroy()
    return dev_type


@cache
def device_matrix_type(*, warn: bool = True) -> str | None:
    r"""Get device matrix type

    Attempt to initialise a GPU device and return the PETSc mat_type
    compatible with that device, or None if no device is detected.
    Typical Usage Example:
    mat_type = device_matrix_type(pc.comm.rank == 0)

    Parameters
    ----------
    warn
        Emit a warning containing the reason a device mat_type
        has not been returned. Defaults to True.

    Raises
    ------
    UnrecognisedDeviceError
        Raised when PETSc initialises a GPU device that
        Firedrake does not understand

    Returns
    -------
    str | None
        The PETSc mat_type compatible with the GPU device detected on
        this system or None

    """
    _device_mat_type_map = {"HOST": None, "CUDA": "aijcusparse", "HIP": "aijhipsparse"}
    dev_type = get_device_type()
    if dev_type is None:
        if warn:
            warnings.warn(
                "This installation of Firedrake is GPU-enabled, but no GPU device has been detected"
            )
        return None
    if dev_type not in _device_mat_type_map:
        raise UnrecognisedDeviceError(
            f"Unknown device type: {dev_type} initialised by PETSc. Firedrake "
            f"currently understands {', '.join([k for k in _device_mat_type_map if k != 'HOST'])}"
            "devices"
        )

    if warn:
        if dev_type == "HOST":
            warnings.warn(
                "This installation of Firedrake is not GPU-enabled, to enable GPU functionality "
                "PETSc will need to be rebuilt with some GPU capability appropriate for this system "
                "(e.g. '--with-cuda=1')."
            )
    return _device_mat_type_map[dev_type]


def _new_uid(comm):
    uid = comm.Get_attr(FIREDRAKE_UID)
    if uid is None:
        uid = 0
    comm.Set_attr(FIREDRAKE_UID, uid + 1)
    return uid


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

def check_netgen_installed() -> None:
    """Check that netgen and ngsPETSc are available.

    If they are not an import error is raised.

    """
    try:
        import netgen  # noqa: F401
        import ngsPETSc  # noqa: F401
    except ImportError:
        raise ImportError(
            "Unable to import netgen and ngsPETSc. Please ensure that they "
            "are installed and available to Firedrake (see "
            "https://www.firedrakeproject.org/install.html#netgen)."
        )


def cached_property_until(key: Callable[[Self], Hashable]):
    """Decorator for a property that is cached until some value changes.

    For example, the ``expensive_property`` below will be cached until
    ``self.value`` changes, and will be recomputed with the new ``self.value``
    and cached when accessed again.

    .. code-block:: python

        class MyClass:

            def __init__(self, value):
                self.value = value

            @cached_property_until(lambda self: self.value)
            def expensive_property(self):
                # Some expensive computation that depends on self.value
                ...
    """
    def decorator(func):
        @property
        @functools.wraps(func)
        def wrapper(self):
            cache_attribute = f"_{func.__name__}_cache"
            current_value = key(self)
            cached_value = getattr(self, cache_attribute, None)
            if cached_value is None or cached_value[0] != current_value:
                result = func(self)
                setattr(self, cache_attribute, (current_value, result))
                return result
            return cached_value[1]
        return wrapper
    return decorator
