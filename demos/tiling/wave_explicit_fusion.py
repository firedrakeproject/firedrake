from firedrake import *

from time import time

from pyop2.profiling import timed_region
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain

from utils.benchmarking import parser, output_time


# Get the input
args = parser()
num_unroll = args.num_unroll
tile_size = args.tile_size
mesh_size = args.mesh_size
verbose = args.verbose
output = args.output
mode = args.fusion_mode

# Constants
loop_chain_length = 3

mesh = UnitSquareMesh(mesh_size, mesh_size)
mesh.init(s_depth=1)
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.
# mesh = Mesh('/tmp/newmeshes/spacefilling1.node', reorder=False)

slope(mesh, debug=True)

# Remove trace bound to avoid running inspections over and over
configuration['lazy_max_trace_length'] = 0
# Switch on PyOP2 profiling
configuration['profiling'] = True

T = 1
dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 1)
p = Function(fs)
phi = Function(fs)

u = TrialFunction(fs)
v = TestFunction(fs)

# Mass lumping
Ml = assemble(v * dx)
Ml.dat._force_evaluation()

p.interpolate(Expression("exp(-40*((x[0]-.5)*(x[0]-.5)+(x[1]-.5)*(x[1]-.5)))"))

if output:
    outfile = File("out.pvd")
    phifile = File("phi.pvd")

while t <= T:
    with loop_chain("main", tile_size=tile_size, num_unroll=num_unroll, mode=mode):
        phi -= dt / 2 * p

        asm = assemble(dt * inner(nabla_grad(v), nabla_grad(phi)) * dx)

        p += asm / Ml

        phi -= dt / 2 * p

        t += dt

# Force evaluation of the PyOP2 trace
start = time()
with timed_region("Time stepping"):
    phi.dat._force_evaluation()
end = time()
print phi.dat.data

# Print runtime summary
output_time(start, end, verbose=verbose, tofile=True, fs=fs, nloops=loop_chain_length,
            tile_size=tile_size)

if output:
    outfile << p
    phifile << phi
