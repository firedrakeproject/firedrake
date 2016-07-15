from firedrake import *

import sys
from os import listdir, environ
from os.path import isfile, join
import mpi4py


def get_S_U_dofs(mesh, p):
    S = TensorFunctionSpace(mesh, 'DG', p, name='S')
    U = VectorFunctionSpace(mesh, 'DG', p, name='U')
    S_tot_dofs = op2.MPI.comm.allreduce(S.dof_count, op=mpi4py.MPI.SUM)
    U_tot_dofs = op2.MPI.comm.allreduce(U.dof_count, op=mpi4py.MPI.SUM)
    return S_tot_dofs, U_tot_dofs


def print_info(mesh, sd, p, cells, U_tot_dofs, S_tot_dofs, nprocs):
    if op2.MPI.comm.rank == 0:
        print info % {
            'mesh': mesh,
            'sd': sd,
            'p': p,
            'cells': cells,
            'U': U_tot_dofs,
            'S': S_tot_dofs,
            'Unp': U_tot_dofs / nprocs,
            'Snp': S_tot_dofs / nprocs
        }
        sys.stdout.flush()
    op2.MPI.comm.barrier()


expected = "<mesh, sd, p, Nelements, U, S, U/np, S/np>"
info = "<%(mesh)22s, %(sd)3d, %(p)3d, %(cells)8d, %(U)8d, %(S)8d, %(Unp)8d, %(Snp)8d >"

poly = [1, 2, 3, 4]
s_depths = [1]

if len(sys.argv) > 1 and sys.args[1] == "noweak":
    nprocs = op2.MPI.comm.size

#################################


if op2.MPI.comm.rank == 0:
    sys.stdout.flush()
    print "Printing info for RectangleMesh (%s):" % expected

Lx, Ly = 300.0, 150.0
all_h = [0.6, 0.45, 0.3, 0.225, 0.15, 0.115]
nprocs = [20, 40, 80, 160, 320, 640]

for p in poly:
    for sd in s_depths:
        for h, n in zip(all_h, nprocs):
            mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
            mesh.topology.init(s_depth=sd)
            S_tot_dofs, U_tot_dofs = get_S_U_dofs(mesh, p)
            print_info(str((Lx, Ly, h)), sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs, n)


#################################

# By default, do not test meshes from gmesh

# meshes_dir = join(environ.get('WORK'), 'meshes', 'wave_elastic')
# if op2.MPI.comm.rank == 0:
#     sys.stdout.flush()
#     print "Printing info for UnstructuredMesh in %s (%s):" % (meshes_dir, expected)
# meshes = [f for f in listdir(meshes_dir) if isfile(join(meshes_dir, f))]

# for p in poly:
#     for sd in s_depths:
#         for m in meshes:
#             mesh = Mesh(join(meshes_dir, m))
#             mesh.topology.init(s_depth=sd)
#             S_tot_dofs, U_tot_dofs = get_S_U_dofs(mesh, p)
#             print_info(m, sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs)
