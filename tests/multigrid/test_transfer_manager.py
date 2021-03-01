import pytest
import numpy
from firedrake import *
from firedrake.mg.ufl_utils import coarsen


@pytest.mark.parametrize("sub", (True, False), ids=["Z.sub(0)", "V"])
@pytest.mark.skipcomplexnoslate
def test_transfer_manager_inside_coarsen(sub):
    mesh = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "N1curl", 2)
    Q = FunctionSpace(mesh, "P", 1)
    Z = V*Q
    x, y = SpatialCoordinate(mesh)

    if sub:
        bc_space = Z.sub(0)
    else:
        bc_space = V
    bcdata = project(as_vector([-y, x]), bc_space)

    bc = DirichletBC(Z.sub(0), bcdata, "on_boundary")

    u = Function(Z)

    v = TestFunction(Z)

    F = inner(u, v)*dx

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem)

    with dmhooks.add_hooks(Z.dm, solver, appctx=solver._ctx):
        cctx = coarsen(solver._ctx, coarsen)

    bc, = cctx._problem.bcs
    V = bc.function_space()
    mesh = V.ufl_domain()
    x, y = SpatialCoordinate(mesh)
    expect = project(as_vector([-y, x]), V)
    assert numpy.allclose(bc.function_arg.dat.data_ro, expect.dat.data_ro)
