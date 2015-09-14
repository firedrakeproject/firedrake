import sys, os

from pyop2.mpi import MPI
from pyop2.profiling import summary

def output_time(start, end, **kwargs):

    verbose = kwargs.get('verbose', False)
    tofile = kwargs.get('tofile', False)

    if MPI.comm.rank in range(1, MPI.comm.size):
        MPI.comm.isend([start, end], dest=0)
    elif MPI.comm.rank == 0:
        starts, ends = [0]*MPI.comm.size, [0]*MPI.comm.size
        starts[0], ends[0] = start, end
        for i in range(1, MPI.comm.size):
            starts[i], ends[i] = MPI.comm.recv(source=i)
        print "MPI starts: %s" % str(starts)
        print "MPI ends: %s" % str(ends)
        start, end = min(starts), max(ends)
        print "Time stepping: ", end - start

    if tofile:
        filename = sys.argv[0][:-3]  # Cut away the extension
        num_threads = os.environ.get("OMP_NUM_THREADS", 1)
        filename = "times/%s/mesh_np%d_nt%d.txt" % (filename, MPI.comm.size, num_threads)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "a") as f:
            f.write("%d : %s" % (MPI.comm.size, end - start))

    if verbose:
        print "Num procs:", MPI.comm.size
        for i in range(MPI.comm.size):
            if MPI.comm.rank == i:
                summary()
            MPI.comm.barrier()
