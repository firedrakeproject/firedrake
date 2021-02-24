from pyop2.mpi import MPI
from itertools import zip_longest

__all__ = ("Ensemble", )


class Ensemble(object):
    def __init__(self, comm, M):
        """
        Create a set of space and ensemble subcommunicators.

        :arg comm: The communicator to split.
        :arg M: the size of the communicators used for spatial parallelism.
        :raises ValueError: if ``M`` does not divide ``comm.size`` exactly.
        """
        size = comm.size

        if (size // M)*M != size:
            raise ValueError("Invalid size of subcommunicators %d does not divide %d" % (M, size))

        rank = comm.rank

        self.global_comm = comm
        """The global communicator."""

        self.comm = comm.Split(color=(rank // M), key=rank)
        """The communicator for spatial parallelism, contains a
        contiguous chunk of M processes from :attr:`comm`"""

        self.ensemble_comm = comm.Split(color=(rank % M), key=rank)
        """The communicator for ensemble parallelism, contains all
        processes in :attr:`comm` which have the same rank in
        :attr:`comm`."""

        assert self.comm.size == M
        assert self.ensemble_comm.size == (size // M)

    def allreduce(self, f, f_reduced, op=MPI.SUM):
        """
        Allreduce a function f into f_reduced over :attr:`ensemble_comm`.

        :arg f: The a :class:`.Function` to allreduce.
        :arg f_reduced: the result of the reduction.
        :arg op: MPI reduction operator.
        :raises ValueError: if communicators mismatch, or function sizes mismatch.
        """
        if MPI.Comm.Compare(f_reduced.comm, f.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Mismatching communicators for functions")
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
            if vout.getSizes() != vin.getSizes():
                raise ValueError("Mismatching sizes")
            self.ensemble_comm.Allreduce(vin.array_r, vout.array, op=op)
        return f_reduced

    def __del__(self):
        if hasattr(self, "comm"):
            self.comm.Free()
            del self.comm
        if hasattr(self, "ensemble_comm"):
            self.ensemble_comm.Free()
            del self.ensemble_comm

    def send(self, f, dest, tag=0):
        """
        Send (blocking) a function f over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg f: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg tag: the tag of the message
        """
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        for dat in f.dat:
            self.ensemble_comm.Send(dat.data_ro, dest=dest, tag=tag)

    def recv(self, f, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, statuses=None):
        """
        Receive (blocking) a function f over :attr:`ensemble_comm` from
        another ensemble rank.

        :arg f: The a :class:`.Function` to receive into
        :arg source: the rank to receive from
        :arg tag: the tag of the message
        :arg statuses: MPI.Status objects (one for each of f.split() or None).
        """
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        if statuses is not None and len(statuses) != len(f.dat):
            raise ValueError("Need to provide enough status objects for all parts of the Function")
        for dat, status in zip_longest(f.dat, statuses or (), fillvalue=None):
            self.ensemble_comm.Recv(dat.data, source=source, tag=tag, status=status)

    def isend(self, f, dest, tag=0):
        """
        Send (non-blocking) a function f over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg f: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg tag: the tag of the message
        :returns: list of MPI.Request objects (one for each of f.split()).
        """
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        return [self.ensemble_comm.Isend(dat.data_ro, dest=dest, tag=tag)
                for dat in f.dat]

    def irecv(self, f, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        """
        Receive (non-blocking) a function f over :attr:`ensemble_comm` from
        another ensemble rank.

        :arg f: The a :class:`.Function` to receive into
        :arg source: the rank to receive from
        :arg tag: the tag of the message
        :returns: list of MPI.Request objects (one for each of f.split()).
        """
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        return [self.ensemble_comm.Irecv(dat.data, source=source, tag=tag)
                for dat in f.dat]
