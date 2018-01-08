from pyop2.mpi import COMM_WORLD

class Subcommunicators(object):
    def __init__(self, N, M):
        """
        Create a set of space and time subcommunicators.

        :arg N: The number of ranks in the time parallelism.
        :arg M: The number of ranks in the space parallelism.
        """
        size = COMM_WORLD.size
        rank = COMM_WORLD.rank
        assert size == M*N

        ### Processes with the same color end up in the same sub communicator
        ### The key is just used to provide a rank order in the sub communicator.
        ## This makes contiguous chunks of size M.
        # use this communicator to instantiate meshes
        self.space_comm = COMM_WORLD.Split(color=(rank // M), key=rank)
        self.space_rank = self.space_comm.rank
        self.time_rank = rank // M

        ## This groups all the processes in the space communicators
        ## that have matching rank. These will correspond to the same bit of
        ## the mesh (hence the same slice through the Vec).
        self.time_comm = COMM_WORLD.Split(color=self.space_comm.rank, key=rank)

    def time_allreduce(self, f, f_reduced):
        """
        Allreduce a function f into f_reduced using the time subcommunicators.
        
        :arg f: The function to allreduce.
        :arg f_reduced: the result of the reduction.
        """

        with f_reduced.dat.vec_wo as vout:
            vout.set(0)
            with f.dat.vec_ro as vin:
                self.time_comm.Allreduce(vin.array_r, vout.array)
