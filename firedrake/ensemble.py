from pyop2.mpi import MPI

__all__ = ("CommManager", )


class CommManager(object):
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

        self.comm = comm
        """The global communicator."""

        self.scomm = comm.Split(color=(rank // M), key=rank)
        """The communicator for spatial parallelism, contains a
        contiguous chunk of M processes from :attr:`comm`"""

        self.ecomm = comm.Split(color=(rank % M), key=rank)
        """The communicator for ensemble parallelism, contains all
        processes in :attr:`comm` which have the same rank in
        :attr:`scomm`."""

        assert self.scomm.size == M
        assert self.ecomm.size == (size // M)

    def allreduce(self, f, f_reduced, op=MPI.SUM):
        """
        Allreduce a function f into f_reduced over :attr:`ecomm`.

        :arg f: The function to allreduce.
        :arg f_reduced: the result of the reduction.
        :arg op: MPI reduction operator.
        :raises ValueError: if communicators mismatch, or function sizes mismatch.
        """
        if MPI.Comm.Compare(f_reduced.comm, f.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Mismatching communicators for functions")
        if MPI.Comm.Compare(f.comm, self.scomm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")
        with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
            if vout.getSizes() != vin.getSizes():
                raise ValueError("Mismatching sizes")
            vout.set(0)
            self.ecomm.Allreduce(vin.array_r, vout.array, op=op)
        return f_reduced

    def __del__(self):
        if hasattr(self, "scomm"):
            self.scomm.Free()
            del self.scomm
        if hasattr(self, "ecomm"):
            self.ecomm.Free()
            del self.ecomm
