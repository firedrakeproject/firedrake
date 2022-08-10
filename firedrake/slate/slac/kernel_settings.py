"""In this file two counters are provided, which are used to ensure unique naming.
   loopy_kernel_counter is used for naming kernels,
   loopy_indexset_counter is used for naming the indices within one kernel.
   Whenever the counter are accessed through their counter functions,
   they are increased within the same step.
"""

from pyop2.mpi import COMM_WORLD
if COMM_WORLD.rank == 0:
    loopy_kernel_counter = COMM_WORLD.bcast(0, root=0)
    loopy_indexset_counter = COMM_WORLD.bcast(0, root=0)
else:
    loopy_kernel_counter = COMM_WORLD.bcast(None, root=0)
    loopy_indexset_counter = COMM_WORLD.bcast(None, root=0)

def knl_counter():
    global loopy_kernel_counter
    c = loopy_kernel_counter
    COMM_WORLD.Barrier()
    if COMM_WORLD.rank == 0:
        loopy_kernel_counter = COMM_WORLD.bcast(c+1, root=0)
    else:
        loopy_kernel_counter = COMM_WORLD.bcast(None, root=0)
    return c


def indexset_counter():
    global loopy_indexset_counter
    c = loopy_indexset_counter
    COMM_WORLD.Barrier()
    if COMM_WORLD.rank == 0:
        loopy_indexset_counter = COMM_WORLD.bcast(c+1, root=0)
    else:
        loopy_indexset_counter = COMM_WORLD.bcast(None, root=0)
    return c
