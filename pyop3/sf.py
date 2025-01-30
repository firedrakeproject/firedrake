from functools import cached_property

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.dtypes import get_mpi_dtype
from pyop3.mpi import internal_comm
from pyop3.utils import just_one


class BufferSizeMismatchException(Exception):
    pass


class StarForest:
    """Convenience wrapper for a `petsc4py.SF`."""

    def __init__(self, sf, size: int):
        self.sf = sf
        self.size = size

        # don't like this pattern
        self._comm = internal_comm(sf.comm, self)

    @classmethod
    def from_graph(cls, size: int, nroots: int, ilocal, iremote, comm):
        # from pyop3.extras.debug import print_with_rank
        # print_with_rank(nroots, ilocal, iremote)
        sf = PETSc.SF().create(comm)
        sf.setGraph(nroots, ilocal, iremote)
        return cls(sf, size)

    @property
    def comm(self):
        return self.sf.comm

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
        return self._graph[0]

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
        return self._graph[1]

    @property
    def iremote(self):
        return self._graph[2]

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

    @cached_property
    def _graph(self):
        return self.sf.getGraph()

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


def single_star(comm, size=1, root=0):
    """Construct a star forest containing a single star.

    The single star has leaves on all ranks apart from the "root" rank that
    point to the same shared data. This is useful for describing globally
    consistent data structures.

    """
    if comm.rank == root:
        # there are no leaves on the root process
        nroots = size
        ilocal = []
        iremote = []
    else:
        nroots = 0
        ilocal = np.arange(size, dtype=np.int32)
        iremote = [(root, i) for i in ilocal]
    return StarForest.from_graph(size, nroots, ilocal, iremote, comm)


def serial_forest(size: int) -> StarForest:
    nroots = 0
    ilocal = []
    iremote = []
    return StarForest.from_graph(size, nroots, ilocal, iremote, MPI.COMM_SELF)
