import math

from pyop2.mpi import MPI


def calculate_sdepth(num_solves, num_unroll, extra_halo):
    """Automate the computation of s_depth based on the input parameters.
    The returned value is an overestimation; that is, it ensures correctness,
    but a smaller s_depth could actually be sufficient.

    :arg num_solves: number of solves per loop chain iteration
    :arg num_unroll: unroll factor for the loop chain
    :arg extra_halo: to expose the nonexec region to the tiling engine
    """
    if MPI.parallel and num_unroll > 0:
        return (int(math.ceil(num_solves/2.0)) or 1) + extra_halo
    else:
        return 1
