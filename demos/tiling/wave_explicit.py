from firedrake import *

from time import time
import cProfile
import os

from pyop2.configuration import configuration
from pyop2.fusion import loop_chain
from pyop2.mpi import MPI

from utils.benchmarking import parser, output_time
from utils.tiling import calculate_sdepth


# Problem constants
num_solves = 1
loop_chain_length = 3


def gen_mesh(args):
    num_unroll = args.num_unroll
    mesh_file = args.mesh_file
    mesh_size = int(args.mesh_size)
    extra_halo = args.extra_halo
    debug_mode = args.debug
    part_mode = args.part_mode
    tile_size = args.tile_size

    kwargs = {'s_depth': calculate_sdepth(num_solves, num_unroll, extra_halo)}
    if part_mode == 'metis':
        kwargs['reorder'] = ('parmetis', tile_size)
    mesh = Mesh(mesh_file) if mesh_file else UnitSquareMesh(mesh_size, mesh_size)
    mesh.topology.init(**kwargs)
    slope(mesh, debug=debug_mode)
    return mesh


def run(args, mesh, time_scale=1):
    num_unroll = args.num_unroll
    tile_size = args.tile_size
    verbose = args.verbose
    output = args.output
    mode = args.fusion_mode
    part_mode = args.part_mode
    extra_halo = args.extra_halo

    # Switch on PyOP2 profiling
    configuration['profiling'] = True

    T = 1 / time_scale
    dt = 0.001
    t = 0
    fs = FunctionSpace(mesh, 'Lagrange', 1, name='fs')
    p = Function(fs)
    phi = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)

    # Mass lumping
    Ml = assemble(v * dx)
    Ml.dat._force_evaluation()

    p.interpolate(Expression("exp(-40*((x[0]-.5)*(x[0]-.5)+(x[1]-.5)*(x[1]-.5)))"))

    # The main form in the timestepping loop
    form = dt * inner(nabla_grad(v), nabla_grad(phi)) * dx

    if output:
        outfile = File("out.pvd")
        phifile = File("phi.pvd")

    start = time()
    while t <= T:
        with loop_chain("main", tile_size=tile_size, num_unroll=num_unroll, mode=mode,
                        extra_halo=extra_halo):
            phi -= dt / 2 * p

            asm = assemble(form)

            p += asm / Ml

            phi -= dt / 2 * p

            t += dt
    end = time()

    print phi.dat.data

    # Print runtime summary
    output_time(start, end,
                verbose=verbose,
                tofile=True,
                fs=fs,
                nloops=loop_chain_length * num_unroll,
                partitioning=part_mode,
                tile_size=tile_size,
                extra_halo=extra_halo,
                poly_order=1,
                domain=os.path.splitext(os.path.basename(mesh.name))[0])

    if output:
        outfile << p
        phifile << phi


if __name__ == '__main__':
    from ffc.log import set_level
    set_level('ERROR')
    args = parser(profile=False)

    mesh = gen_mesh(args)
    if args.profile:
        try:
            location = os.path.join(os.environ['FIREDRAKE_DIR'], 'demos', 'tiling')
        except:
            location = '.'
        location = os.path.join(location, 'log_wave_explicit_nu%d_rank%d.cprofile'
                                % (args.num_unroll, MPI.comm.rank))
        time_scale = 20.0
        cProfile.run("""run(args, mesh, time_scale=time_scale)""", location)
    else:
        run(args, mesh)
