from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
import pytest
import numpy


@pytest.fixture(scope='module', params=["div", "curl"])
def problem(request):
    return request.param


@pytest.fixture(scope='module', params=[1, 2, 3])
def degree(request):
    return request.param


sp = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS
}


def error(N, problem, degree):
    mesh = UnitSquareMesh(N, N, quadrilateral=True)

    if problem == "div":
        op = div
        family = "BDMCF"
    elif problem == "curl":
        op = curl
        family = "BDMCE"
    else:
        raise ValueError

    V = FunctionSpace(mesh, family, degree)
    u = Function(V)
    v = TestFunction(V)

    (x, y) = SpatialCoordinate(mesh)
    u_ex = as_vector([sin(2*pi*x) * cos(2*pi*y),
                      x * (1-x) * y * (1-y)])

    if problem == "div":
        f = u_ex - grad(div(u_ex))
    else:
        f = u_ex + curl(curl(u_ex))

    F = inner(op(u), op(v))*dx + inner(u, v)*dx - inner(f, v)*dx
    bc = DirichletBC(V, project(u_ex, V, solver_parameters=sp), "on_boundary")

    solve(F == 0, u, bc, solver_parameters=sp)

    err = errornorm(u_ex, u, "L2")
    return err


def test_bdmc_riesz_map(problem, degree):

    errors = []
    for N in [10, 20, 40]:
        errors.append(error(N, problem, degree))

    convergence_orders = lambda x: numpy.log2(numpy.array(x)[:-1] / numpy.array(x)[1:])
    conv = convergence_orders(errors)
    print("errors: ", errors)
    print("convergence order: ", conv)

    tol = 0.11

    assert (conv > (degree + 1) - tol).all()
