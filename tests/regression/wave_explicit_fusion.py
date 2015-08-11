# NOTE This is a demo, not a regression test

from firedrake import *

import sys
from time import time

from pyop2.profiling import summary, timed_region, Timer, tic, toc
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.base import _trace
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

T = 1
dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 1)
p = Function(fs)
phi = Function(fs)

u = TrialFunction(fs)
v = TestFunction(fs)

p.interpolate(Expression("exp(-40*((x[0]-.5)*(x[0]-.5)+(x[1]-.5)*(x[1]-.5)))"))

if output:
    #outfile = File("out.pvd")
    phifile = File("phi.pvd")

# Mass matrix
m = u * v * dx

lump_mass = True

step = 0
while t <= T:
    with loop_chain("main", num_unroll=2, tile_size=2000):
        step += 1

        phi -= dt / 2 * p

        if lump_mass:
            p += (assemble(dt * inner(nabla_grad(v), nabla_grad(phi)) * dx)
                  / assemble(v * dx))
        else:
            solve(u * v * dx == v * p * dx + dt * inner(
                nabla_grad(v), nabla_grad(phi)) * dx, p)

        phi -= dt / 2 * p

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
