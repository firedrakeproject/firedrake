from firedrake import *
import pytest
import numpy


@pytest.fixture(scope='module', params=["div", "curl"])
def problem(request):
    return request.param


# FIXME: MMS does not converge for degree 1
# @pytest.fixture(scope='module', params=[1, 2, 3])
@pytest.fixture(scope='module', params=[2, 3])
def degree(request):
    return request.param


@pytest.fixture(scope='module', params=[2])
def dim(request):
    return request.param


sp = {"snes_type": "ksponly",
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "mat_mumps_icntl_14": 200}


def error(N, problem, degree, dim):
    mesh = UnitSquareMesh(N, N, quadrilateral=True)
    if dim == 3:
        mesh = ExtrudedMesh(mesh, N)

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

    k = 1
    if dim == 2:
        (x, y) = SpatialCoordinate(mesh)
        u_ex = as_vector([sin(2*pi*x) * cos(2*pi*y),
                          x * (1-x) * y * (1-y)])
    else:
        (x, y, z) = SpatialCoordinate(mesh)
        u_ex = as_vector([sin(2*pi*k*x) * cos(2*pi*k*y) * sin(4*pi*z),
                          x * (1-x) * z * (1-z),
                          y * (1-y) * z * (1-z)])

    if problem == "div":
        f = u_ex - grad(div(u_ex))
    else:
        f = u_ex + curl(curl(u_ex))

    F = inner(op(u), op(v))*dx + inner(u, v)*dx - inner(f, v)*dx
    bc = DirichletBC(V, project(u_ex, V, solver_parameters=sp), "on_boundary")

    solve(F == 0, u, bc, solver_parameters=sp)

    err = errornorm(u_ex, u, "L2")
    return err


def test_bdmc_riesz_map(problem, degree, dim):

    if dim == 2:
        Ns = [10, 20, 40]
    else:
        Ns = [2, 4, 8]

    errors = []
    for N in Ns:
        errors.append(error(N, problem, degree, dim))

    convergence_orders = lambda x: numpy.log2(numpy.array(x)[:-1] / numpy.array(x)[1:])
    conv = convergence_orders(errors)
    print("errors: ", errors)
    print("convergence order: ", conv)

    assert numpy.allclose(conv, degree, atol=0.1)
