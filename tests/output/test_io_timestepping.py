import pytest
from firedrake import *
from pyop2.mpi import COMM_WORLD
import ufl
import os

cwd = os.path.abspath(os.path.dirname(__file__))

mesh_name = "channel"
func_name = "f"


def _get_expr(V, i):
    mesh = V.mesh()
    element = V.ufl_element()
    x, y = SpatialCoordinate(mesh)
    shape = element.value_shape()
    if element.family() == "Real":
        return 7. + i * i
    elif shape == ():
        return x * y * (i + 5)
    elif shape == (2, ):
        return as_vector([x * (i + 2), y * (i + 3)])
    else:
        raise NotImplementedError(f"Not testing for shape = {shape}")


def _project(f, expr, method):
    if f.function_space().ufl_element().family() == "Real":
        f.dat.data.itemset(expr)
    elif method == "project":
        getattr(f, method)(expr, solver_parameters={"ksp_rtol": 1.e-16})
    elif method == "interpolate":
        getattr(f, method)(expr)


@pytest.fixture(params=["P1", "BDMF3", "P2-P1", "Real"])
def element(request):
    if request.param == "P1":
        return ufl.FiniteElement("P", ufl.triangle, 1)
    elif request.param == "BDMF3":
        return ufl.FiniteElement("BDMF", ufl.triangle, 3)
    elif request.param == "P2-P1":
        return ufl.MixedElement(ufl.FiniteElement("P", ufl.triangle, 2),
                                ufl.FiniteElement("P", ufl.triangle, 1))
    elif request.param == "Real":
        return ufl.FiniteElement("Real", ufl.triangle, 0)


@pytest.mark.parallel(nprocs=3)
def test_io_timestepping(element, tmpdir):
    filename = os.path.join(str(tmpdir), "test_io_timestepping_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mycolor = (COMM_WORLD.rank > COMM_WORLD.size - 1)
    comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
    method = "project" if isinstance(element, ufl.MixedElement) else "interpolate"
    if mycolor == 0:
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name, comm=comm)
        V = FunctionSpace(mesh, element)
        f = Function(V, name=func_name)
        with CheckpointFile(filename, 'w', comm=comm) as afile:
            for i in range(5):
                _project(f, _get_expr(V, i), method)
                afile.save_function(f, idx=i)
    mycolor = (COMM_WORLD.rank > COMM_WORLD.size - 2)
    comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
    if mycolor == 0:
        with CheckpointFile(filename, 'r', comm=comm) as afile:
            mesh = afile.load_mesh(mesh_name)
            for i in range(5):
                f = afile.load_function(mesh, func_name, idx=i)
                V = f.function_space()
                g = Function(V)
                _project(g, _get_expr(V, i), method)
                assert assemble(inner(g - f, g - f) * dx) < 1.e-16
