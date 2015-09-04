# NOTE This is a demo, not a regression test

from firedrake import *

import numpy as np
import sys
from time import time

from pyop2.profiling import summary, timed_region, Timer
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.mpi import MPI

verbose = True if len(sys.argv) == 2 and sys.argv[1] == '--verbose' else False
output = False

mesh = UnitSquareMesh(10, 10)
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.
# mesh = Mesh('/tmp/newmeshes/spacefilling1.node', reorder=False)

slope(mesh, debug=True)

# Remove trace bound to avoid running inspections over and over
configuration['lazy_max_trace_length'] = 0
# Switch on PyOP2 profiling
configuration['profiling'] = True

print "MPI rank", MPI.comm.rank, "has a Mesh size of", mesh.num_cells(), "cells."

# Simulation
scale = 1.0
N = 100

# 1) Solve -- Setup
V = FunctionSpace(mesh, 'Lagrange', 1)
total_dofs = np.zeros(1, dtype=int)
op2.MPI.comm.Allreduce(np.array([V.dof_dset.size], dtype=int), total_dofs)
if verbose:
    print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size

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
    with loop_chain("main", tile_size=20, num_unroll=1):
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
if MPI.comm.rank in range(1, MPI.comm.size):
    MPI.comm.isend([start, end], dest=0)
elif MPI.comm.rank == 0:
    starts, ends = [0]*MPI.comm.size, [0]*MPI.comm.size
    starts[0], ends[0] = start, end
    for i in range(1, MPI.comm.size):
        starts[i], ends[i] = MPI.comm.recv(source=i)
    print "MPI starts: %s" % str(starts)
    print "MPI ends: %s" % str(ends)
    start = min(starts)
    end = max(ends)
    tot = end - start
    print "Time stepping: ", tot

if verbose:
    print "Num procs:", MPI.comm.size
    for i in range(MPI.comm.size):
        if MPI.comm.rank == i:
            summary()
        MPI.comm.barrier()

    if MPI.comm.rank == 0:
        print "MPI rank", MPI.comm.rank, ":"
        print "  Time stepping loop:", Timer._timers['Time stepping'].total
        print "  ParLoop kernel:", Timer._timers['ParLoop kernel'].total
        if "ParLoopChain: executor" in Timer._timers:
            print "ParLoopChain: compute: ", Timer._timers['ParLoopChain: compute'].total

if output:
    #outfile << p
    phifile << phi
