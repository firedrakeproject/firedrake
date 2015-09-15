import sys
import os
import numpy as np

from pyop2.mpi import MPI
from pyop2.profiling import summary


def output_time(start, end, **kwargs):

    verbose = kwargs.get('verbose', False)
    tofile = kwargs.get('tofile', False)
    fs = kwargs.get('fs', None)
    nloops = kwargs.get('nloops', 0)
    tile_size = kwargs.get('tile_size', 0)

    # Find number of processes, and number of threads per process
    num_procs = MPI.comm.size
    num_threads = os.environ.get("OMP_NUM_THREADS", 1)

    # So what execution /mode/ is this?
    if num_procs == 1 and num_threads == 1:
        modes = ['sequential', 'omp', 'mpi', 'mpi_openmp']
    elif num_procs == 1 and num_threads > 1:
        modes = ['openmp']
    elif num_procs > 1 and num_threads == 1:
        modes = ['mpi']
    else:
        modes = ['mpi_openmp']

    # Find the total execution time
    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend([start, end], dest=0)
    elif MPI.comm.rank == 0:
        starts, ends = [0]*num_procs, [0]*num_procs
        starts[0], ends[0] = start, end
        for i in range(1, num_procs):
            starts[i], ends[i] = MPI.comm.recv(source=i)
        print "MPI starts: %s" % str(starts)
        print "MPI ends: %s" % str(ends)
        start, end = min(starts), max(ends)
        print "Time stepping: ", end - start

    # Find the total mesh size
    mesh_size = 0
    if fs:
        total_dofs = np.zeros(1, dtype=int)
        MPI.comm.Allreduce(np.array([fs.dof_dset.size], dtype=int), total_dofs)
        mesh_size = total_dofs

    # Print to file
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    if MPI.comm.rank == 0 and tofile:
        name = sys.argv[0][:-3]  # Cut away the extension
        for mode in modes:
            filename = "times/%s/mesh%d/%s/np%d_nt%d.txt" % \
                (name, mesh_size, mode, num_procs, num_threads)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            # Read the old content, add the new time value, order
            # everything based on <execution time, #loops tiled>, write
            # back to the file (overwriting existing content)
            with open(filename, "w+") as f:
                lines = [line for line in f if line.strip()]
                lines = [i.split(':').replace(' ', '') for i in lines]
                lines = [(num(i[0]), num(i[1]), num(i[2])) for i in lines]
                lines += [(end-start, nloops, tile_size)]
                lines.sort(key=lambda x: (x[0], -x[1]))
                lines = "\n".join(["%s : %s : %s" % i for i in lines])
                lines += "\n"
                f.write(lines)

    if verbose:
        print "Num procs:", num_procs
        for i in range(num_procs):
            if MPI.comm.rank == i:
                summary()
            MPI.comm.barrier()
