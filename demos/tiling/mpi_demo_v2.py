from firedrake import *

import sys
from time import time
import numpy as np

from pyop2.profiling import summary, timed_region, Timer
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.mpi import MPI
from pyop2.base import _trace

from utils.benchmarking import parser

from ffc.log import set_level


def run(mesh, fs1, fs2, tile_size, num_unroll):
    # print "MPI rank", MPI.comm.rank, "has a Mesh size of", mesh.num_cells(), "cells."
    T = 0.05
    dt = 0.001
    t = 0

    p1 = Function(fs1, dtype=np.dtype('int32'), name="p1").assign(1)
    p2 = Function(fs1, dtype=np.dtype('int32'), name="p2").assign(2)
    p3 = Function(fs2, dtype=np.dtype('int32'), name="p3").assign(0)
    coords = mesh.coordinates

    _trace.evaluate_all()

    time_init_code = """
    for(int i=0; i<B.dofs; i++) {
      for(int j=0; j<C.dofs; j++) {
        B[i][0] += C[j][0] - 2;
      }
    }
    """

    write_code = """
    for(int i=0; i<A.dofs; i++) {
      A[i][0] = A[i][0] + B[i][0];
    }
    """

    incr_code = """
    int tmp = 0;
    for(int j=0; j<A.dofs; j++) {
      tmp += A[j][0];
    }
    for(int i=0; i<C.dofs; i++) {
      C[i][0] += tmp;
    }
    """

    while t < T:
        par_loop(time_init_code, dx, {'B': (p2, INC), 'C': (p3, READ)})
        _trace.evaluate_all()
        with loop_chain("main2", mode=mode, tile_size=tile_size, num_unroll=num_unroll):

            par_loop(write_code, dx, {'A': (p1, RW), 'B': (p2, READ), 'S': (coords, READ)})

            par_loop(incr_code, dS, {'C': (p3, INC), 'A': (p1, READ), 'S': (coords, READ)})

            t += dt

    return p1, p2, p3


set_level('ERROR')

# Parse input
args = parser()
tile_size = args.tile_size
mesh_size = int(args.mesh_size)
mode = args.fusion_mode

np.set_printoptions(precision=13)

# Default mesh
# mesh = UnitSquareMesh(15, 15)
# mesh.topology.init(s_depth=4)
Lx = 300.0
Ly = 150.0
h = 2.5
mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
mesh.topology.init(s_depth=4)
slope(mesh, debug=True)

# Switch on PyOP2 profiling
configuration['profiling'] = True

fs1 = FunctionSpace(mesh, 'DG', 1, name='fs1')
fs2 = FunctionSpace(mesh, 'DG', 2, name='fs2')

p1_orig, p2_orig, p3_orig = run(mesh, fs1, fs2, tile_size, 0)
p1_tile, p2_tile, p3_tile = run(mesh, fs1, fs2, tile_size, 1)

p1_orig = p1_orig.dat._data[:p1_orig.dat.dataset.size]
p2_orig = p2_orig.dat._data[:p2_orig.dat.dataset.size]
p3_orig = p3_orig.dat._data[:p3_orig.dat.dataset.size]
p1_tile = p1_tile.dat._data[:p1_tile.dat.dataset.size]
p2_tile = p2_tile.dat._data[:p2_tile.dat.dataset.size]
p3_tile = p3_tile.dat._data[:p3_tile.dat.dataset.size]

MPI.comm.barrier()
for r in range(MPI.comm.size):
    if MPI.comm.rank == r:
        #print MPI.comm.rank, ": ", p3.dat._data[:p3.dat.dataset.size]
        #print MPI.comm.rank, ": ", np.linalg.norm(p3.dat._data[:p3.dat.dataset.size])
        #print MPI.comm.rank, ": ", sum(p3.dat._data[:p3.dat.dataset.size])
        print MPI.comm.rank, ": ", (p1_orig == p1_tile).all(), (p2_orig == p2_tile).all(), (p3_orig == p3_tile).all()
    MPI.comm.barrier()
