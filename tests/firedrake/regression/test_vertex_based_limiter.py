import pytest
from firedrake import *
import numpy as np


@pytest.fixture(params=["periodic-interval",
                        "periodic-square-tri", "periodic-square-quad"])
def mesh(request):
    if request.param == "periodic-interval":
        return PeriodicUnitIntervalMesh(30)
    elif request.param == "periodic-square-tri":
        return PeriodicUnitSquareMesh(30, 30)
    elif request.param == "periodic-square-quad":
        return PeriodicUnitSquareMesh(30, 30, quadrilateral=True)


def space(m):
    element = BrokenElement(m.coordinates.function_space().ufl_element().sub_elements[0])
    return FunctionSpace(m, element)


@pytest.mark.skipcomplex
def test_constant_field(mesh):
    # test function space
    v = space(mesh)

    # Create limiter
    limiter = VertexBasedLimiter(v)

    # Set up constant field
    u0 = Constant(1)
    u = Function(v).interpolate(u0)
    u_old = Function(u)

    limiter.apply(u)
    diff = assemble(inner(u - u_old, u - u_old) * dx) ** 0.5
    assert diff < 1.0e-10, "Failed on Constant function"


@pytest.mark.skipcomplex
def test_step_function_bounds(mesh):
    x = SpatialCoordinate(mesh)

    # test function space
    v = space(mesh)

    # Create limiter
    limiter = VertexBasedLimiter(v)

    # advecting velocity
    u0 = conditional(real(x[0]) < 0.5, 1., 0.)
    u = Function(v).interpolate(u0)
    limiter.apply(u)

    assert np.max(u.dat.data_ro) <= 1.0, "Failed by exceeding max values"
    assert np.min(u.dat.data_ro) >= 0.0, "Failed by exceeding min values"


@pytest.mark.skipcomplex
def test_step_function_loop(mesh, iterations=100):
    # test function space
    v = space(mesh)
    m = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    if m.shape == (1, ):
        u0 = as_vector([1])
    else:
        u0 = as_vector([1, 0])
    u = Function(m).interpolate(u0)

    # advection problem
    dt = 1. / iterations
    phi = TestFunction(v)
    D = TrialFunction(v)
    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))  # upwind value

    a_mass = inner(D, phi) * dx
    a_int = inner(-u * D, grad(phi)) * dx
    a_flux = inner(un('+') * D('+') - un('-') * D('-'), jump(phi)) * dS
    arhs = a_mass - dt * (a_int + a_flux)

    dD1 = Function(v)
    D1 = Function(v)
    x = SpatialCoordinate(mesh)

    # Initial Conditions
    D0 = conditional(real(x[0]) < 0.5, 1., 0.)

    D = Function(v).interpolate(D0)
    D1.assign(D)

    t = 0.0
    T = iterations * dt
    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})

    # Make slope limiter
    limiter = VertexBasedLimiter(v)
    limiter.apply(D)

    while t < (T - dt / 2):
        D1.assign(D)
        limiter.apply(D1)
        solver.solve()
        D1.assign(dD1)
        limiter.apply(D1)

        solver.solve()
        D1.assign(0.75 * D + 0.25 * dD1)
        limiter.apply(D1)
        solver.solve()
        D.assign((1.0 / 3.0) * D + (2.0 / 3.0) * dD1)
        limiter.apply(D1)

        t += dt

    assert np.max(u.dat.data_ro) <= 1.0, "Failed by exceeding max values"
    assert np.min(u.dat.data_ro) >= 0.0, "Failed by exceeding min values"


@pytest.mark.parallel
@pytest.mark.skipcomplex
def test_parallel_limiting():
    serial_result = _apply_limiter_with_comm(COMM_SELF)
    parallel_result = _apply_limiter_with_comm(COMM_WORLD)
    assert np.allclose(serial_result, parallel_result)


def _apply_limiter_with_comm(comm):
    mesh = RectangleMesh(10, 4, 5000., 1000., comm=comm)
    V = space(mesh)
    f = Function(V)
    x, *_ = SpatialCoordinate(mesh)
    f.project(sin(2*pi*x/3000.))
    limiter = VertexBasedLimiter(V)
    limiter.apply(f)

    return np.asarray([
        norm(f), norm(limiter.centroids), norm(limiter.min_field), norm(limiter.max_field)
    ])
