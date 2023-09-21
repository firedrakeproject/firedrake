import pytest
from firedrake import *
from pyop2.mpi import COMM_WORLD
import os

cwd = os.path.abspath(os.path.dirname(__file__))

mesh_name = "channel"
func_name = "f"
labelVal = 4


def _solve_poisson(msh):
    V = FunctionSpace(msh, "CG", 3)
    x, y = SpatialCoordinate(msh)
    f = Function(V).interpolate(x * x * y)
    v = TestFunction(V)
    u = TrialFunction(V)
    a = inner(u, v) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(V, Constant(1., domain=msh), labelVal)
    sol = Function(V, name=func_name)
    solve(a == L, sol, bcs=[bc, ])
    return sol


@pytest.mark.parallel(nprocs=7)
def test_io_solve_poisson(tmpdir):
    filename = os.path.join(str(tmpdir), "test_io_solve_poisson_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mycolor = (COMM_WORLD.rank > COMM_WORLD.size - 1)
    comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
    if mycolor == 0:
        msh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name, comm=comm)
        solA = _solve_poisson(msh)
        with CheckpointFile(filename, 'w', comm=comm) as afile:
            afile.save_function(solA)
    mycolor = (COMM_WORLD.rank > COMM_WORLD.size - 2)
    comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
    if mycolor == 0:
        with CheckpointFile(filename, 'r', comm=comm) as afile:
            msh = afile.load_mesh(mesh_name)
            solA = afile.load_function(msh, func_name)
        solB = _solve_poisson(msh)
        assert assemble(inner(solB - solA, solB - solA) * dx) < 1.e-16
