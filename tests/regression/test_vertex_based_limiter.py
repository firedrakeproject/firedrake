import pytest
from firedrake import *
import numpy as np


@pytest.fixture(params=["periodic-square-tri"])
def mesh(request):
    if request.param == "periodic-interval":
        return PeriodicUnitIntervalMesh(30)
    elif request.param == "periodic-square-tri":
        return UnitSquareMesh(3, 3)
    elif request.param == "periodic-square-quad":
        return PeriodicUnitSquareMesh(3, 3, quadrilateral=True)


@pytest.fixture(params=["P1DG", "PDG2"])
def degree(request):
    if request.param == "P1DG":
        return 1
    if request.param == "PDG2":
        return 2


def test_constant_field(mesh, degree):
    # test function space
    v = FunctionSpace(mesh, "DG", degree)

    # Create limiter
    limiter = KuzminLimiter(v)

    # Set up constant field
    u0 = Constant(1)
    u = Function(v).interpolate(u0)
    u_old = Function(u)

    limiter.apply(u)
    diff = assemble((u - u_old) ** 2 * dx) ** 0.5
    print diff
    assert diff < 1.0e-10, "Failed on Constant function"


def test_step_function_bounds(mesh, degree):
    x = SpatialCoordinate(mesh)
    file = File("nameful2.pvd")

    # test function space
    v = FunctionSpace(mesh, "DG", degree)

    # Create limiter
    limiter = KuzminLimiter(v)

    # Generate step function
    u0 = conditional(x[0] < 0.5, 1., 0.)
    u = Function(v).interpolate(u0)
    u_old = Function(u)
    file.write(u)

    limiter.apply(u)
    file.write(u)
    diff1 = assemble((u - u_old) ** 2 * dx) ** 0.5
    limiter.apply(u)
    file.write(u)
    diff = assemble((u - u_old) ** 2 * dx) ** 0.5
    print "diffs", diff1, diff
    i_max = np.argmax(u.dat.data_ro)
    i_min = np.argmin(u.dat.data_ro)
    s = np.sort(u.dat.data_ro)
    print "mins", s[:80]
    print "min:", u.dat.data_ro[i_min], "max:",  u.dat.data_ro[i_max]
    assert u.dat.data_ro[i_max] <= 1.0 + 1e-10, "Failed by exceeding max values"
    assert u.dat.data_ro[i_min] >= 0.0 - 1e-10, "Failed by exceeding min values"


def test_step_function_loop(mesh, degree, iterations=100):
    # test function space
    v = FunctionSpace(mesh, "DG", degree)
    m = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    if m.shape == (1, ):
        u0 = as_vector([1])
    else:
        u0 = as_vector([1, 0])
    u = Function(m).interpolate(u0)
    print "here"

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
    limiter.apply(D1)

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

    print len(D1.dat.data) / 6
    diff = assemble((D1 - D1_old) ** 2 * dx) ** 0.5
    print "Error:", diff
    max = np.max(D1.dat.data_ro)
    min = np.min(D1.dat.data_ro)
    print "Max:", max, "Min:", min
    assert max <= 1.0 + 1e-2, "Failed by exceeding max values"
    assert min >= 0.0 - 1e-2, "Failed by exceeding min values"


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
