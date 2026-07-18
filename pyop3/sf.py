from __future__ import annotations

import abc
import dataclasses
import numbers
import typing
from functools import cached_property
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.record
from pyop3 import utils
from pyop3.dtypes import get_mpi_dtype, IntType


if typing.TYPE_CHECKING:
    from pyop3.axis_tree import AxisComponentRegionSizeT


from ._sf_cy import (  # noqa: F401
    filter_petsc_sf,
    create_petsc_section_sf,
    renumber_petsc_sf,
    mask_petsc_sf,
)


class BufferSizeMismatchException(Exception):
    pass


class AbstractStarForest(abc.ABC):

    # {{{ abstract methods

    @classmethod
    @abc.abstractmethod
    def merge(cls, sfs) -> Self:
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __eq__(self, other: Any, /) -> bool:
        pass

    # @property
    # @abc.abstractmethod
    # def iroot(self) -> np.ndarray:
    #     pass

    @property
    @abc.abstractmethod
    def ileaf(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def num_owned(self) -> AxisComponentRegionSizeT:
        pass

    @property
    @abc.abstractmethod
    def num_ghost(self) -> AxisComponentRegionSizeT:
        pass

    @abc.abstractmethod
    def broadcast_begin(self, *args):
        pass

    @abc.abstractmethod
    def broadcast_end(self, *args):
        pass

    # }}}

    def broadcast(self, *args):
        self.broadcast_begin(*args)
        self.broadcast_end(*args)


def _check_poison(func):
    def wrapper(self, *args):
        assert not self._poisoned
        return func(self, *args)
    return wrapper


@pyop3.record.record()
class StarForest(AbstractStarForest):
    """Convenience wrapper for a `petsc4py.SF`."""

    # {{{ instance attrs

    sf: PETSc.SF
    _comm: MPI.Comm
    _poisoned: bool = False
    """Debugging attribute, turn exchanges into errors."""
    # only for root values, bcasting is fine I think

    # def __init__(self, sf, comm, mask=None):
    #     self._orig_sf = sf
    #
    #     if mask is not None:
    #         sf = mask_petsc_sf(sf, mask)
    #     self.sf = sf
    #     self.mask = mask
    #     self._comm = comm

    # }}}

    # {{{ factory methods

    @classmethod
    def from_graph(cls, size: IntType, ilocal, iremote, comm):
        size = utils.strict_int(size)
        ilocal = ilocal.astype(IntType, casting="safe")
        iremote = iremote.astype(IntType, casting="safe")

        sf = PETSc.SF().create(comm)
        sf.setGraph(size, ilocal, iremote)
        return cls(sf, comm)

    @classmethod
    def merge(cls, sfs) -> Self:
        assert all(isinstance(sf, cls) for sf in sfs)

        if len(sfs) == 1:
            return utils.just_one(sfs)

        size = sum(sf.size for sf in sfs)
        ilocal = np.concatenate([sf.ilocal for sf in sfs])
        iremote = np.concatenate([sf.iremote for sf in sfs])
        comm = utils.single_valued(sf.comm for sf in sfs)
        return cls.from_graph(size, ilocal, iremote, comm)

    # }}}

    # {{{ interface impls

    comm = pyop3.record.attr("_comm")

    def __hash__(self) -> int:
        return hash((
            type(self),
            # self.nroots,  # this isn't a meaningful attr
            self.ilocal.data.tobytes(),
            self.iremote.data.tobytes(),
        ))

    def __eq__(self, /, other: Any) -> bool:
        return (
            type(other) is type(self)
            # and other.nroots == self.nroots  # this isn't a meaningful attr
            and (other.ilocal == self.ilocal).all()
            and (other.iremote == self.iremote).all()
        )

    # @cached_property
    # def iroot(self):
    #     """Return the indices of roots on the current process."""
    #     # mark leaves and reduce
    #     mask = np.full(self.size, False, dtype=bool)
    #     mask[self.ileaf] = True
    #     self.reduce(mask, MPI.REPLACE)
    #
    #     # now clear the leaf indices, the remaining marked indices are roots
    #     mask[self.ileaf] = False
    #     return utils.just_one(np.nonzero(mask))

    @property
    def ileaf(self):
        return self.ilocal

    # @cached_property
    # def icore(self):
    #     """Return the indices of points that are not roots or leaves."""
    #     mask = np.full(self.size, True, dtype=bool)
    #     mask[self.iroot] = False
    #     mask[self.ileaf] = False
    #     return utils.just_one(np.nonzero(mask))

    @property
    def num_owned(self):
        num_owned =  self.size - self.nleaves
        assert num_owned >= 0
        return num_owned

    @property
    def nleaves(self):
        return len(self.ileaf)



    # }}}

    @property
    def size(self):
        return self.graph[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.sf}, {self.size})"

    # better alias?
    @property
    def num_ghost(self):
        return self.nleaves

    @property
    def ilocal(self):
        return self.graph[1]

    @property
    def iremote(self):
        return self.graph[2]

    @property
    def graph(self):
        return self.sf.getGraph()

    # @_check_poison
    def broadcast(self, *args):
        self.broadcast_begin(*args)
        self.broadcast_end(*args)

    # @_check_poison
    def broadcast_begin(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastBegin(*bcast_args)

    # @_check_poison
    def broadcast_end(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastEnd(*bcast_args)

    @_check_poison
    def reduce(self, *args):
        self.reduce_begin(*args)
        self.reduce_end(*args)

    @_check_poison
    def reduce_begin(self, *args):
        reduce_args = self._prepare_args(*args)
        self.sf.reduceBegin(*reduce_args)

    @_check_poison
    def reduce_end(self, *args):
        reduce_args = self._prepare_args(*args)
        self.sf.reduceEnd(*reduce_args)

    def _prepare_args(self, *args):
        if len(args) == 3:
            from_buffer, to_buffer, op = args
        elif len(args) == 2:
            from_buffer, op = args
            to_buffer = from_buffer
        else:
            raise ValueError

        if any(len(buf) != self.size for buf in [from_buffer, to_buffer]):
            raise BufferSizeMismatchException

        # what about cdim?
        dtype, _ = get_mpi_dtype(from_buffer.dtype)
        return (dtype, from_buffer, to_buffer, op)

    def with_section(self, section: PETSc.Section) -> Self:
        """Create a new star forest via composition with a PETSc section."""
        petsc_sf = create_petsc_section_sf(self.sf, section)
        return type(self)(petsc_sf, self.comm)

    def filter(self, indices) -> Self:
        petsc_sf = filter_petsc_sf(self.sf, indices, 0, self.size)
        return type(self)(petsc_sf, self.comm)


# FIXME: Do we really need to have a size attr?
class NullStarForest(AbstractStarForest):

    # {{{ instance attrs

    def __init__(self, size):
        self.size = size
        self.__post_init__()

    def __post_init__(self):
        # for ragged not true
        # assert isinstance(self.size, numbers.Integral)
        pass

    # }}}

    # {{{ factory methods

    @classmethod
    def merge(cls, sfs) -> Self:
        assert all(isinstance(sf, cls) for sf in sfs)
        size = sum(sf.size for sf in sfs)
        return cls(size)

    # {{{ interface impls

    def __hash__(self) -> int:
        return hash((type(self), self.size))

    def __eq__(self, /, other: Any) -> bool:
        return type(other) is type(self) and other.size == self.size

    @property
    def ileaf(self):
        return np.empty(0, dtype=IntType)

    @property
    def num_owned(self) -> AxisComponentRegionSizeT:
        return self.size

    @property
    def num_ghost(self) -> int:
        return 0

    def broadcast_begin(self, *args):
        pass

    def broadcast_end(self, *args):
        pass

    # }}}

    def __repr__(self, /) -> str:
        return f"NullStarForest({self.size})"

    # TODO: This leads to some very unclear semantics. Basically there are
    # subtle differences between having a null star forest and an SF that is
    # 'None' and sometimes we want to treat them as equivalent and other
    # times not.
    def __bool__(self) -> bool:
        return False

    @property
    def comm(self) -> MPI.Comm:
        return MPI.COMM_SELF

    def reduce_begin(self, *args):
        pass

    def reduce_end(self, *args):
        pass


def single_star_sf(comm: MPI.Comm, size: IntType = IntType.type(1), root: int = 0):
    """Construct a star forest containing a single star.

    The single star has leaves on all ranks apart from the "root" rank that
    point to the same shared data. This is useful for describing globally
    consistent data structures.

    """
    if comm.rank == root:
        # there are no leaves on the root process
        ilocal = np.empty(0, dtype=np.int32)
        iremote = np.empty(0, dtype=np.int32)
    else:
        ilocal = np.arange(size, dtype=np.int32)
        iremote = np.stack([np.full(size, root, dtype=np.int32), ilocal], axis=1)
    return StarForest.from_graph(size, ilocal, iremote, comm)


def local_sf(size: numbers.Integral, comm: MPI.Comm) -> StarForest:
    size = IntType.type(size)
    ilocal = np.empty(0, dtype=IntType)
    iremote = np.empty(0, dtype=IntType)
    return StarForest.from_graph(size, ilocal, iremote, comm)
