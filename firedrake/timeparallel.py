from pyop2.mpi import MPI

__all__ = ("CommManager", )


class CommManager(object):
    def __init__(self, comm, M):
        """
        Create a set of space and time subcommunicators.

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

        self.tcomm = comm.Split(color=(rank % M), key=rank)
        """The communicator for time parallelism, contains all
        processes in :attr:`comm` which have the same rank in
        :attr:`scomm`."""

        assert self.scomm.size == M
        assert self.tcomm.size == (size // M)

    def allreduce(self, f, f_reduced, op=MPI.SUM):
        """
        Allreduce a function f into f_reduced over :attr:`tcomm`.

        :arg f: The function to allreduce.
        :arg f_reduced: the result of the reduction.
        :arg op: MPI reduction operator.
        :raises ValueError: if communicators mismatch, or function sizes mismatch.
        """
        if MPI.Group.Compare(f_reduced.comm.Get_group(), f.comm.Get_group()) != MPI.IDENT:
            raise ValueError("Mismatching communicators for functions")
        if MPI.Group.Compare(f.comm.Get_group(), self.scomm.Get_group()) != MPI.IDENT:
            raise ValueError("Function communicator does not match space communicator")
        with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
            if vout.getSizes() != vin.getSizes():
                raise ValueError("Mismatching sizes")
            vout.set(0)
            self.tcomm.Allreduce(vin.array_r, vout.array, op=op)
        return f_reduced

    def __del__(self):
        if hasattr(self, "scomm"):
            self.scomm.Free()
            del self.scomm
        if hasattr(self, "tcomm"):
            self.tcomm.Free()
            del self.tcomm
