# NOTE This is a demo, not a regression test

from firedrake import *

import numpy as np
import sys
from time import time

from pyop2.profiling import timed_region
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.mpi import MPI

from utils.timing import output_time

verbose = False
output = False

# Constants
loop_chain_length = 4

# Parameters
num_unroll = 1
tile_size = 20

mesh = UnitSquareMesh(10, 10)
mesh.init(s_depth=1)
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.
# mesh = Mesh('/tmp/newmeshes/spacefilling1.node', reorder=False)

slope(mesh, debug=True)

# Remove trace bound to avoid running inspections over and over
configuration['lazy_max_trace_length'] = 0
# Switch on PyOP2 profiling
configuration['profiling'] = True

# Simulation
scale = 1.0
N = 100

# 1) Solve -- Setup
V = FunctionSpace(mesh, 'Lagrange', 1)

p = Function(V)
phi = Function(V, name="phi")

u = TrialFunction(V)
v = TestFunction(V)

bcval = Constant(0.0)
bc = DirichletBC(V, bcval, 1)

# Mass lumping
Ml = assemble(1.0 / assemble(v*dx))

dt = 0.001 * scale
dtc = Constant(dt)
t = 0.0

rhs = inner(grad(v), grad(phi)) * dx

if output:
    # Initial value of /phi/
    phifile = File("vtk/firedrake_wave_%s.pvd" % scale)
    phifile << phi

b = assemble(rhs)
dphi = 0.5 * dtc * p
dp = dtc * Ml * b

# 2) Solve -- Timestepping
while t < N*dt:
    with loop_chain("main", tile_size=tile_size, num_unroll=num_unroll):
        print "Executing timestep ", t
        bcval.assign(sin(2*pi*5*t))

        phi -= dphi

        # Using mass lumping to solve
        assemble(rhs, tensor=b)
        p += dp
        bc.apply(p)

        phi -= dphi

        t += dt

# Force evaluation of the PyOP2 trace
start = time()
with timed_region("Time stepping"):
    phi.dat._force_evaluation()
end = time()

# Print runtime summary
output_time(start, end, verbose=verbose, tofile=True, fs=V, nloops=loop_chain_length,
            tile_size=tile_size)

if output:
    #outfile << p
    phifile << phi
