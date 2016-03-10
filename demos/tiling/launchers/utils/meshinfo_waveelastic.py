from firedrake import *

from os import listdir, environ
from os.path import isfile, join
import mpi4py


def get_S_U_dofs(mesh, p):
    S = TensorFunctionSpace(mesh, 'DG', p, name='S')
    U = VectorFunctionSpace(mesh, 'DG', p, name='U')
    S_tot_dofs = op2.MPI.comm.allreduce(S.dof_count, op=mpi4py.MPI.SUM)
    U_tot_dofs = op2.MPI.comm.allreduce(U.dof_count, op=mpi4py.MPI.SUM)
    return S_tot_dofs, U_tot_dofs


def print_info(mesh, sd, p, cells, U_tot_dofs, S_tot_dofs):
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
    op2.MPI.comm.barrier()


expected = "<mesh,sd,p,Nelements,U,S,U/np,S/np>"
info = "    <%(mesh)s, %(sd)d, %(p)d, %(cells)d, %(U)d, %(S)d, %(Unp)d, %(Snp)d>"

poly = [1, 2, 3]
s_depths = [1, 2, 3, 4]

nprocs = op2.MPI.comm.size

#################################


print "Printing info for RectangleMesh (%s):" % expected
Lx, Ly = 300.0, 150.0
all_h = [2.5, 2.0, 1.0, 0.8, 0.4, 0.2]

for p in poly:
    for sd in s_depths:
        for h in all_h:
            mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
            mesh.topology.init(s_depth=sd)
            S_tot_dofs, U_tot_dofs = get_S_U_dofs(mesh, p)
            print_info(str((Lx, Ly, h)), sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs)


#################################

meshes_dir = join(environ.get('WORK'), 'meshes', 'wave_elastic')
print "Printing info for UnstructuredMesh in %s (%s):" % (meshes_dir, expected)
meshes = [f for f in listdir(meshes_dir) if isfile(join(meshes_dir, f))]

for p in poly:
    for sd in s_depths:
        for m in meshes:
            mesh = Mesh(join(meshes_dir, m))
            mesh.topology.init(s_depth=sd)
            S_tot_dofs, U_tot_dofs = get_S_U_dofs(mesh, p)
            print_info(m, sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs)
