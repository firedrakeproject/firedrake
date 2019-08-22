# Some generic python utilities not really specific to our work.
from decorator import decorator
from pyop2.utils import cached_property  # noqa: F401
from mpi4py import MPI
from pyop2.mpi import refcount_keyval
from itertools import count


uid_keyval = MPI.Comm.Create_keyval()


def _new_uid(comm):
    """Produce a new unique ID on this communicator."""
    if comm.Get_attr(refcount_keyval) is None:
        raise ValueError("Must be called on an internal communicator")
    uid = comm.Get_attr(uid_keyval)
    if uid is None:
        uid = count()
        comm.Set_attr(uid_keyval, uid)
    return next(uid)


def _init():
    """Cause :func:`pyop2.init` to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    :func:`pyop2.init` if she wants to set a non-default option, for example
    to switch the debug or log level."""
    from pyop2 import op2
    from firedrake.parameters import parameters
    if not op2.initialised():
        op2.init(**parameters["pyop2_options"])


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
