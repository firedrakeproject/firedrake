import numpy as np
from collections import defaultdict
from mpi4py import MPI

from pyop2 import op2

import utils


class Halo(object):
    """Build a Halo associated with the appropriate FunctionSpace.

    The Halo is derived from a PetscSF object and builds the global
    to universal numbering map from the respective PetscSections."""

    def __init__(self, petscsf, global_numbering, universal_numbering, vdim):
        self._tag = utils._new_uid()
        self._comm = op2.MPI.comm
        self._nprocs = self.comm.size
        self._sends = defaultdict(list)
        self._receives = defaultdict(list)
        self._gnn2unn = None
        self._sf = petscsf.duplicate()
        remote_sends = defaultdict(list)

        if op2.MPI.comm.size <= 1:
            return

        # Sort the SF by local indices
        nroots, nleaves, local, remote = petscsf.getGraph()
        local_new, remote_new = (list(x) for x in zip(*sorted(zip(local, remote), key=lambda x: x[0])))
        petscsf.setGraph(nroots, nleaves, local_new, remote_new)

        # Derive local receives and according remote sends
        nroots, nleaves, local, remote = petscsf.getGraph()
        for local, (rank, index) in zip(local, remote):
            if rank != self.comm.rank:
                self._receives[rank].append(local/vdim)
                remote_sends[rank].append(index/vdim)

        # Propagate remote send lists to the actual sender
        send_reqs = []
        for p in range(self._nprocs):
            # send sizes
            if p != self._comm.rank:
                s = np.array(len(remote_sends[p]), dtype=np.int32)
                send_reqs.append(self.comm.Isend(s, dest=p, tag=self.tag))

        recv_reqs = []
        sizes = [np.empty(1, dtype=np.int32) for _ in range(self._nprocs)]
        for p in range(self._nprocs):
            # receive sizes
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(sizes[p], source=p, tag=self.tag))

        MPI.Request.Waitall(recv_reqs)
        MPI.Request.Waitall(send_reqs)

        for p in range(self._nprocs):
            # allocate buffers
            if p != self._comm.rank:
                self._sends[p] = np.empty(sizes[p], dtype=np.int32)

        send_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                send_buf = np.array(remote_sends[p], dtype=np.int32)
                send_reqs.append(self.comm.Isend(send_buf, dest=p, tag=self.tag))

        recv_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(self._sends[p], source=p, tag=self.tag))

        MPI.Request.Waitall(send_reqs)
        MPI.Request.Waitall(recv_reqs)

        # Build Global-To-Universal mapping
        pStart, pEnd = global_numbering.getChart()
        self._gnn2unn = np.zeros(global_numbering.getStorageSize()/vdim, dtype=np.int32)
        for p in range(pStart, pEnd):
            dof = global_numbering.getDof(p) / vdim
            goff = global_numbering.getOffset(p) / vdim
            uoff = universal_numbering.getOffset(p) / vdim
            if uoff < 0:
                uoff = (-1*uoff)-1
            for c in range(dof):
                self._gnn2unn[goff+c] = uoff+c

    @utils.cached_property
    def op2_halo(self):
        if not self.sends and not self.receives:
            return None
        return op2.Halo(self.sends, self.receives,
                        comm=self.comm, gnn2unn=self.gnn2unn)

    @property
    def comm(self):
        return self._comm

    @property
    def tag(self):
        return self._tag

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def sends(self):
        return self._sends

    @property
    def receives(self):
        return self._receives

    @property
    def gnn2unn(self):
        return self._gnn2unn
