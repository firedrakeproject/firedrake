from __future__ import annotations

import abc
import numbers
import typing
from functools import cached_property
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.dtypes import get_mpi_dtype, IntType
from pyop3.mpi import internal_comm
from pyop3.utils import just_one, strict_int


if typing.TYPE_CHECKING:
    from pyop3.tree.axis_tree import AxisComponentRegionSizeT


from ._sf_cy import filter_petsc_sf, create_petsc_section_sf, renumber_petsc_sf  # noqa: F401


# This is so we can more easily distinguish internal and external comms
# It is still necessary to register weakref finalizers for these (see what
# we do in Firedrake).
class Pyop3Comm(MPI.Comm):
    pass


class ParallelAwareObject(abc.ABC):
    """Abstract class for objects that know about communicators.

    Unlike `DistributedObject`s, it is allowed for objects inheriting from
    this class to have `None` for communicator values.

    """

    @property
    @abc.abstractmethod
    def user_comm(self) -> MPI.Comm | None:
        pass

    # TODO: probably decorate as 'collective'
    @property
    def internal_comm(self) -> Pyop3Comm | None:
        if self.user_comm is None:
            return None

        # this is where the magic happens...
        # but not yet
        return self.user_comm

    # TODO: cast to a Pyop3 and register a weakref handler of some kind
    @staticmethod
    def register_comm(self, comm) -> Pyop3Comm:
        pass


class DistributedObject(ParallelAwareObject, metaclass=abc.ABCMeta):
    """Abstract class for objects that have a parallel execution context.

    The expected usage is for classes to implement the attribute `user_comm`.

    """

    @property
    def internal_comm(self) -> Pyop3Comm:
        # this is where the magic happens...
        # but not yet
        assert self.user_comm is not None
        return self.user_comm

    @property
    @abc.abstractmethod
    def user_comm(self) -> MPI.Comm:
        pass


class BufferSizeMismatchException(Exception):
    pass


class AbstractStarForest(DistributedObject, abc.ABC):

    # {{{ abstract methods

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __eq__(self, other: Any, /) -> bool:
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

    def __init__(self, size: AxisComponentRegionSizeT) -> None:
        self.size = size

    def broadcast(self, *args):
        self.broadcast_begin(*args)
        self.broadcast_end(*args)



class StarForest(AbstractStarForest):
    """Convenience wrapper for a `petsc4py.SF`."""

    # {{{ interface impls

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

    # }}}

    # NOTE: I think 'size' now has to equal the number of roots
    def __init__(self, sf: PETSc.SF, size: IntType) -> None:
        size = strict_int(size)

        # TODO: This check only makes sense for SFs that we make (where ghosts are at the end)
        # in the general case it isn't true
        # _check_sf(sf)

        num_roots, local_leaf_indices, _ = sf.getGraph()
        assert size >= num_roots and size >= len(local_leaf_indices)

        self.sf = sf
        super().__init__(size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.sf}, {self.size})"

    @classmethod
    def from_graph(cls, size: IntType, nroots: IntType, ilocal, iremote, comm):
        size = strict_int(size)
        ilocal = ilocal.astype(IntType, casting="safe")
        iremote = iremote.astype(IntType, casting="safe")

        sf = PETSc.SF().create(comm)
        sf.setGraph(nroots, ilocal, iremote)
        return cls(sf, size)

    @property
    def user_comm(self) -> MPI.Comm:
        return self.sf.comm.tompi4py()

    @cached_property
    def iroot(self):
        """Return the indices of roots on the current process."""
        # mark leaves and reduce
        mask = np.full(self.size, False, dtype=bool)
        mask[self.ileaf] = True
        self.reduce(mask, MPI.REPLACE)

        # now clear the leaf indices, the remaining marked indices are roots
        mask[self.ileaf] = False
        return just_one(np.nonzero(mask))

    @property
    def ileaf(self):
        return self.ilocal

    @cached_property
    def icore(self):
        """Return the indices of points that are not roots or leaves."""
        mask = np.full(self.size, True, dtype=bool)
        mask[self.iroot] = False
        mask[self.ileaf] = False
        return just_one(np.nonzero(mask))

    # not useful
    # @property
    # def nroots(self):
    #     return self.graph[0]

    @property
    def nowned(self):
        num_owned =  self.size - self.nleaves
        assert num_owned >= 0
        return num_owned

    # better alias
    @property
    def num_owned(self):
        return self.nowned

    @property
    def nleaves(self):
        return len(self.ileaf)

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

    def broadcast_begin(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastBegin(*bcast_args)

    def broadcast_end(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastEnd(*bcast_args)

    def reduce(self, *args):
        self.reduce_begin(*args)
        self.reduce_end(*args)

    def reduce_begin(self, *args):
        reduce_args = self._prepare_args(*args)
        self.sf.reduceBegin(*reduce_args)

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
            breakpoint()
            raise BufferSizeMismatchException

        # what about cdim?
        dtype, _ = get_mpi_dtype(from_buffer.dtype)
        return (dtype, from_buffer, to_buffer, op)


class NullStarForest(AbstractStarForest):

    # {{{ interface impls

    def __hash__(self) -> int:
        return hash((type(self), self.size))

    def __eq__(self, /, other: Any) -> bool:
        return type(other) is type(self) and other.size == self.size

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

    # TODO: This leads to some very unclear semantics. Basically there are
    # subtle differences between having a null star forest and an SF that is
    # 'None' and sometimes we want to treat them as equivalent and other
    # times not.
    def __bool__(self) -> bool:
        return False

    @property
    def user_comm(self) -> MPI.Comm:
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
        nroots = size
        # there are no leaves on the root process
        ilocal = np.empty(0, dtype=np.int32)
        iremote = np.empty(0, dtype=np.int32)
    else:
        nroots = 0
        ilocal = np.arange(size, dtype=np.int32)
        iremote = np.stack([np.full(size, root, dtype=np.int32), ilocal], axis=1)
    return StarForest.from_graph(size, nroots, ilocal, iremote, comm)


def local_sf(size: numbers.Integral, comm: MPI.Comm) -> StarForest:
    # nroots = IntType.type(0)
    nroots = IntType.type(size)
    ilocal = np.empty(0, dtype=IntType)
    iremote = np.empty(0, dtype=IntType)
    return StarForest.from_graph(size, nroots, ilocal, iremote, comm)


def _check_sf(sf: PETSc.SF):
    # sanity check: leaves should always be at the end of the array
    size, leaf_indices, _ = sf.getGraph()
    num_leaves = len(leaf_indices)
    assert (leaf_indices == np.arange(size-num_leaves, size, dtype=IntType)).all()
