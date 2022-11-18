from pyop2.mpi import MPI, internal_comm, decref
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

        # User global comm
        self.global_comm = comm
        # Internal global comm
        self._global_comm = internal_comm(comm)

        # User split comm
        self.comm = self.global_comm.Split(color=(rank // M), key=rank)
        # Internal split comm
        self._comm = internal_comm(self.comm)
        """The communicator for spatial parallelism, contains a
        contiguous chunk of M processes from :attr:`comm`"""

        # User ensemble comm
        self.ensemble_comm = self.global_comm.Split(color=(rank % M), key=rank)
        # Internal ensemble comm
        self._ensemble_comm = internal_comm(self.ensemble_comm)
        """The communicator for ensemble parallelism, contains all
        processes in :attr:`comm` which have the same rank in
        :attr:`comm`."""

        assert self.comm.size == M
        assert self.ensemble_comm.size == (size // M)

    def __del__(self):
        if hasattr(self, "comm"):
            self.comm.Free()
            del self.comm
        if hasattr(self, "ensemble_comm"):
            self.ensemble_comm.Free()
            del self.ensemble_comm
        for comm_name in ["_global_comm", "_comm", "_ensemble_comm"]:
            if hasattr(self, comm_name):
                comm = getattr(self, comm_name)
                decref(comm)

    def _check_function(self, f, g=None):
        """
        Check if function f (and possibly a second function g) is a valid argument for ensemble mpi routines

        :arg f: The function to check
        :arg g: Second function to check
        :raises ValueError: if function communicators mismatch each other or the ensemble spatial communicator, or is the functions are in different spaces
        """
        if MPI.Comm.Compare(f._comm, self._comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")

        if g is not None:
            if MPI.Comm.Compare(f._comm, g._comm) not in {MPI.CONGRUENT, MPI.IDENT}:
                raise ValueError("Mismatching communicators for functions")
            if f.function_space() != g.function_space():
                raise ValueError("Mismatching function spaces for functions")

    def allreduce(self, f, f_reduced, op=MPI.SUM):
        """
        Allreduce a function f into f_reduced over :attr:`ensemble_comm`.

        :arg f: The a :class:`.Function` to allreduce.
        :arg f_reduced: the result of the reduction.
        :arg op: MPI reduction operator.
        :raises ValueError: if function communicators mismatch each other or the ensemble spatial communicator, or if the functions are in different spaces
        """
        self._check_function(f, f_reduced)

        with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
            self.ensemble_comm.Allreduce(vin.array_r, vout.array, op=op)
        return f_reduced

    def iallreduce(self, f, f_reduced, op=MPI.SUM):
        """
        Allreduce (non-blocking) a function f into f_reduced over :attr:`ensemble_comm`.

        :arg f: The a :class:`.Function` to allreduce.
        :arg f_reduced: the result of the reduction.
        :arg op: MPI reduction operator.
        :returns: list of MPI.Request objects (one for each of f.split()).
        :raises ValueError: if function communicators mismatch each other or the ensemble spatial communicator, or if the functions are in different spaces
        """
        self._check_function(f, f_reduced)

        return [self.ensemble_comm.Iallreduce(fdat.data, rdat.data, op=op)
                for fdat, rdat in zip(f.dat, f_reduced.dat)]

    def reduce(self, f, f_reduced, op=MPI.SUM, root=0):
        """
        Reduce a function f into f_reduced over :attr:`ensemble_comm` to rank root

        :arg f: The a :class:`.Function` to reduce.
        :arg f_reduced: the result of the reduction on rank root.
        :arg op: MPI reduction operator.
        :arg root: rank to reduce to
        :raises ValueError: if function communicators mismatch each other or the ensemble spatial communicator, or is the functions are in different spaces
        """
        self._check_function(f, f_reduced)

        # need to use `vec` not `vec_wo` for f_reduced otherwise function will be blanked out
        # when rank!=root because `vec_wo` doesn't copy over existing data into the pyop2 vector
        with f_reduced.dat.vec as vout, f.dat.vec_ro as vin:
            self.ensemble_comm.Reduce(vin.array_r, vout.array, op=op, root=root)

        return f_reduced

    def ireduce(self, f, f_reduced, op=MPI.SUM, root=0):
        """
        Reduce (non-blocking) a function f into f_reduced over :attr:`ensemble_comm` to rank root

        :arg f: The a :class:`.Function` to reduce.
        :arg f_reduced: the result of the reduction on rank root.
        :arg op: MPI reduction operator.
        :arg root: rank to reduce to
        :returns: list of MPI.Request objects (one for each of f.split()).
        :raises ValueError: if function communicators mismatch each other or the ensemble spatial communicator, or is the functions are in different spaces
        """
        self._check_function(f, f_reduced)

        return [self.ensemble_comm.Ireduce(fdat.data, rdat.data, op=op, root=root)
                for fdat, rdat in zip(f.dat, f_reduced.dat)]

    def bcast(self, f, root=0):
        """
        Broadcast a function f over :attr:`ensemble_comm` from rank root

        :arg f: The :class:`.Function` to broadcast.
        :arg root: rank to broadcast from
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)

        with f.dat.vec as vec:
            self.ensemble_comm.Bcast(vec.array, root=root)
        return f

    def ibcast(self, f, root=0):
        """
        Broadcast (non-blocking) a function f over :attr:`ensemble_comm` from rank root

        :arg f: The :class:`.Function` to broadcast.
        :arg root: rank to broadcast from
        :returns: list of MPI.Request objects (one for each of f.split()).
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)

        return [self.ensemble_comm.Ibcast(dat.data, root=root)
                for dat in f.dat]

    def send(self, f, dest, tag=0):
        """
        Send (blocking) a function f over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg f: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg tag: the tag of the message
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)
        for dat in f.dat:
            self._ensemble_comm.Send(dat.data_ro, dest=dest, tag=tag)

    def recv(self, f, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, statuses=None):
        """
        Receive (blocking) a function f over :attr:`ensemble_comm` from
        another ensemble rank.

        :arg f: The a :class:`.Function` to receive into
        :arg source: the rank to receive from
        :arg tag: the tag of the message
        :arg statuses: MPI.Status objects (one for each of f.split() or None).
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)
        if statuses is not None and len(statuses) != len(f.dat):
            raise ValueError("Need to provide enough status objects for all parts of the Function")
        for dat, status in zip_longest(f.dat, statuses or (), fillvalue=None):
            self._ensemble_comm.Recv(dat.data, source=source, tag=tag, status=status)

    def isend(self, f, dest, tag=0):
        """
        Send (non-blocking) a function f over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg f: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg tag: the tag of the message
        :returns: list of MPI.Request objects (one for each of f.split()).
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)
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
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        self._check_function(f)
        return [self.ensemble_comm.Irecv(dat.data, source=source, tag=tag)
                for dat in f.dat]

    def sendrecv(self, fsend, dest, sendtag=0, frecv=None, source=MPI.ANY_SOURCE, recvtag=MPI.ANY_TAG, status=None):
        """
        Send (blocking) a function fsend and receive a function frecv over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg fsend: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg sendtag: the tag of the send message
        :arg frecv: The a :class:`.Function` to receive into
        :arg source: the rank to receive from
        :arg recvtag: the tag of the received message
        :arg status: MPI.Status object or None.
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        # functions don't necessarily have to match
        self._check_function(fsend)
        self._check_function(frecv)
        with fsend.dat.vec_ro as sendvec, frecv.dat.vec_wo as recvvec:
            self.ensemble_comm.Sendrecv(sendvec, dest, sendtag=sendtag,
                                        recvbuf=recvvec, source=source, recvtag=recvtag,
                                        status=status)

    def isendrecv(self, fsend, dest, sendtag=0, frecv=None, source=MPI.ANY_SOURCE, recvtag=MPI.ANY_TAG):
        """
        Send a function fsend and receive a function frecv over :attr:`ensemble_comm` to another
        ensemble rank.

        :arg fsend: The a :class:`.Function` to send
        :arg dest: the rank to send to
        :arg sendtag: the tag of the send message
        :arg frecv: The a :class:`.Function` to receive into
        :arg source: the rank to receive from
        :arg recvtag: the tag of the received message
        :returns: list of MPI.Request objects (one for each of fsend.split() and frecv.split()).
        :raises ValueError: if function communicator mismatches the ensemble spatial communicator.
        """
        # functions don't necessarily have to match
        self._check_function(fsend)
        self._check_function(frecv)
        requests = []
        requests.extend([self.ensemble_comm.Isend(dat.data_ro, dest=dest, tag=sendtag) for dat in fsend.dat])
        requests.extend([self.ensemble_comm.Irecv(dat.data, source=source, tag=recvtag) for dat in frecv.dat])
        return requests
