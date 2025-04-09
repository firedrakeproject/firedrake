from functools import cached_property

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.dtypes import get_mpi_dtype, IntType
from pyop2.mpi import internal_comm
from pyop3.utils import just_one, strict_int


class BufferSizeMismatchException(Exception):
    pass


class StarForest:
    """Convenience wrapper for a `petsc4py.SF`."""

    def __init__(self, sf, size: IntType):
        self.sf = sf
        self.size = strict_int(size)

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
    def comm(self) -> MPI.Comm:
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

    @property
    def nroots(self):
        return self.graph[0]

    @property
    def nowned(self):
        return self.size - self.nleaves

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

    def broadcast(self, *args):
        self.broadcast_begin(*args)
        self.broadcast_end(*args)

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
            raise BufferSizeMismatchException

        # what about cdim?
        dtype, _ = get_mpi_dtype(from_buffer.dtype)
        return (dtype, from_buffer, to_buffer, op)


def single_star_sf(comm, size=1, root=0):
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
        iremote = [(root, i) for i in ilocal]
    return StarForest.from_graph(size, nroots, ilocal, iremote, comm)


def local_sf(size: IntType, comm: MPI.Comm) -> StarForest:
    nroots = IntType(0)
    ilocal = np.empty(0, dtype=IntType)
    iremote = np.empty(0, dtype=IntType)
    return StarForest.from_graph(size, nroots, ilocal, iremote, comm)
