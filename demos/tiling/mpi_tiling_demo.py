from firedrake import *

import sys
from time import time
import numpy as np

from pyop2.profiling import summary, timed_region, Timer
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.mpi import MPI
from pyop2.base import _trace

verbose = True if len(sys.argv) == 2 and sys.argv[1] == '--verbose' else False
output = False

np.set_printoptions(precision=13)

mesh = UnitSquareMesh(10, 10)
mesh.init(s_depth=2)
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.
# mesh = Mesh('/tmp/newmeshes/spacefilling1.node', reorder=False)

slope(mesh, debug=True)

# Remove trace bound to avoid running inspections over and over
configuration['lazy_max_trace_length'] = 0
# Switch on PyOP2 profiling
configuration['profiling'] = True

print "MPI rank", MPI.comm.rank, "has a Mesh size of", mesh.num_cells(), "cells."

T = 1.000
dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 1)
p = Function(fs, dtype=np.dtype('int32')).assign(1)

_trace.evaluate_all()

incr_code = """
for(int i=0; i<A.dofs; i++) {
  A[i][0] += 1;
}
"""

times2_code = """
  A[0] *= 2 - 1;
"""

while t <= T:
    with loop_chain("main", tile_size=5, num_unroll=0):
        par_loop(times2_code, direct, {'A': (p, RW)})

        par_loop(incr_code, dx, {'A': (p, INC)})

        par_loop(times2_code, direct, {'A': (p, RW)})

        t += dt

_trace.evaluate_all()
print p.dat._data[:p.dat.dataset.size]
