from firedrake import *

from time import time

from pyop2.profiling import timed_region
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain

from utils.benchmarking import parser, output_time
from utils.tiling import calculate_sdepth


# Get the input
args = parser()
num_unroll = args.num_unroll
tile_size = args.tile_size
mesh_file = args.mesh_file
mesh_size = int(args.mesh_size)
verbose = args.verbose
output = args.output
mode = args.fusion_mode
part_mode = args.part_mode
extra_halo = args.extra_halo
debug_mode = args.debug

# Sanity check of the input
assert num_unroll in [0, 1], "Cannot run with unroll factor > 1 (bcs not in trace)"

# Constants
loop_chain_length = 4
num_solves = 1

mesh = Mesh(mesh_file) if mesh_file else UnitSquareMesh(mesh_size, mesh_size)
mesh.init(s_depth=calculate_sdepth(num_solves, num_unroll, extra_halo))
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.
# mesh = Mesh('/tmp/newmeshes/spacefilling1.node', reorder=False)

slope(mesh, debug=debug_mode)

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
start = time()
while t < N*dt:
    with loop_chain("main", tile_size=tile_size, num_unroll=num_unroll, mode=mode,
                    partitioning=part_mode, extra_halo=extra_halo):
        print "Executing timestep ", t
        bcval.assign(sin(2*pi*5*t))

        phi -= dphi

        # Using mass lumping to solve
        assemble(rhs, tensor=b)
        p += dp
        bc.apply(p)

        phi -= dphi

        t += dt
end = time()

# Print runtime summary
output_time(start, end,
            verbose=verbose,
            tofile=True,
            fs=V,
            nloops=loop_chain_length * num_unroll,
            partitioning=part_mode,
            tile_size=tile_size)

if output:
    #outfile << p
    phifile << phi
