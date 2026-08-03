import collections
import functools
import numbers
import types
from collections.abc import Iterable
from typing import Any

import loopy as lp
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.obj
import pyop3.sf


@functools.singledispatch
def get_comm(obj: Any, /) -> MPI.Comm:
    """Return the communicator associated with an object.

    If no communicator is available (e.g. trying to get the comm of an integer)
    then ``COMM_SELF`` is used.

    """
    utils.raise_missing_dispatch_handler(obj)


@get_comm.register
def _(comm: MPI.Comm, /) -> MPI.Comm:
    return comm


@get_comm.register
def _(obj: PETSc.Object, /) -> MPI.Comm:
    raise TypeError("Cannot get the right comm off of a PETSc object, you just get the internal PETSc one")


@get_comm.register
def _(sf: pyop3.sf.AbstractStarForest, /) -> MPI.Comm:
    return sf.comm


@get_comm.register
def _(obj: pyop3.obj.Object, /) -> MPI.Comm:
    return obj.comm


@get_comm.register(str)
@get_comm.register(numbers.Number)
@get_comm.register(types.NoneType)
@get_comm.register(lp.TranslationUnit)
@get_comm.register(pyop3.constants.Intent)
@get_comm.register(pyop3.constants._Decide)  # pyop3.DECIDE
def _(_, /) -> MPI.Comm:
    return MPI.COMM_SELF


@get_comm.register
def _(iterable: tuple | list, /) -> MPI.Comm:
    return common_comm(iterable, default=MPI.COMM_SELF)


@get_comm.register
def _(mapping: collections.abc.Mapping, /) -> MPI.Comm:
    return common_comm(mapping.values(), default=MPI.COMM_SELF)


@get_comm.register
def _(set_: collections.abc.Set, /) -> MPI.Comm:
    assert all(get_comm(item) == MPI.COMM_SELF for item in set_), \
        "Cannot have parallelism inside a set (unordered)"
    return MPI.COMM_SELF


def common_comm(objects: Iterable[Any], **kwargs) -> MPI.Comm:
    """Return a communicator valid for all objects.

    The valid communicator is defined as the one with the largest size.

    Parameters
    ----------
    objects
        Communicator-carrying objects to inspect. All object must define
        a ``comm`` attribute.

    Returns
    -------
    MPI.Comm
        A communicator that the provided objects are safely collective over.

    """
    return pyop3.mpi.common_comm(map(get_comm, objects), **kwargs)


def single_comm(*objects: Iterable[Any]) -> MPI.Comm:
    """Return the single comm shared by all objects."""
    return utils.single_valued(map(get_comm, objects))
