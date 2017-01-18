import pytest
from firedrake import *
import numpy as np


@pytest.fixture(params=["periodic-interval1", "periodic-interval2",
                        "periodic-square-tri1", "periodic-square-quad1"])
def mesh_degree(request):
    if request.param == "periodic-interval1":
        return (PeriodicUnitIntervalMesh(30), 1)
    if request.param == "periodic-interval2":
        return (PeriodicUnitIntervalMesh(30), 2)
    elif request.param == "periodic-square-tri1":
        return (PeriodicUnitSquareMesh(30, 30), 1)
    elif request.param == "periodic-square-quad1":
        return (PeriodicUnitSquareMesh(30, 30, quadrilateral=True), 1)


def test_constant_field(mesh_degree):
    # test function space
    mesh, degree = mesh_degree
    v = FunctionSpace(mesh, "DG", degree)

    # Create limiter
    limiter = KuzminLimiter(v)

    # Set up constant field
    u0 = Constant(1)
    u = Function(v).interpolate(u0)
    u_old = Function(u)

    limiter.apply(u)
    diff = assemble((u - u_old) ** 2 * dx) ** 0.5
    assert diff < 1.0e-10, "Failed on Constant function"
    assert np.max(u.dat.data_ro) <= 1.0 + 1e-10, "Failed by exceeding max values"
    assert np.min(u.dat.data_ro) >= 0.0 - 1e-10, "Failed by exceeding min values"


def test_step_function_bounds(mesh_degree):
    mesh, degree = mesh_degree
    x = SpatialCoordinate(mesh)

    # test function space
    v = FunctionSpace(mesh, "DG", degree)

    # Create limiter
    limiter = KuzminLimiter(v)

    # Generate step function
    u0 = conditional(x[0] < 0.5, 1., 0.)
    u = Function(v).interpolate(u0)

    limiter.apply(u)
    assert np.max(u.dat.data_ro) <= 1.0 + 1e-10, "Failed by exceeding max values"
    assert np.min(u.dat.data_ro) >= 0.0 - 1e-10, "Failed by exceeding min values"


def test_step_function_loop(mesh_degree, iterations=100):
    mesh, degree = mesh_degree
    # test function space
    v = FunctionSpace(mesh, "DG", degree)
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

    a_mass = phi * D * dx
    a_int = dot(grad(phi), -u * D) * dx
    a_flux = dot(jump(phi), un('+') * D('+') - un('-') * D('-')) * dS
    arhs = a_mass - dt * (a_int + a_flux)

    dD1 = Function(v)
    D1 = Function(v)
    x = SpatialCoordinate(mesh)

    # Initial Conditions
    D0 = conditional(x[0] > 0.5, 1, 0.)

    D = Function(v).interpolate(D0)
    D1.assign(D)
    D1_old = Function(D1)

    t = 0.0
    T = iterations * dt
    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})

    # Make slope limiter
    limiter = KuzminLimiter(v)

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
        limiter.apply(D)

        t += dt

    diff = assemble((D1 - D1_old) ** 2 * dx) ** 0.5
    max = np.max(D1.dat.data_ro)
    min = np.min(D1.dat.data_ro)
    print "Max:", max, "Min:", min
    print diff
    assert max <= 1.0 + 1e-2, "Failed by exceeding max values"
    assert min >= 0.0 - 1e-2, "Failed by exceeding min values"


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
