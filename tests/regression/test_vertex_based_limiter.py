import pytest
from firedrake import *
import numpy as np
import subprocess
import sys


@pytest.fixture(params=["periodic-interval",
                        "periodic-square-tri", "periodic-square-quad"])
def mesh(request):
    if request.param == "periodic-interval":
        return PeriodicUnitIntervalMesh(30)
    elif request.param == "periodic-square-tri":
        return PeriodicUnitSquareMesh(30, 30)
    elif request.param == "periodic-square-quad":
        return PeriodicUnitSquareMesh(30, 30, quadrilateral=True)


def test_constant_field(mesh):
    # test function space
    v = FunctionSpace(mesh, "DG", 1)

    # Create limiter
    limiter = VertexBasedLimiter(v)

    # Set up constant field
    u0 = Constant(1)
    u = Function(v).interpolate(u0)
    u_old = Function(u)

    limiter.apply(u)
    diff = assemble((u - u_old) ** 2 * dx) ** 0.5
    assert diff < 1.0e-10, "Failed on Constant function"


def test_step_function_bounds(mesh):
    x = SpatialCoordinate(mesh)

    # test function space
    v = FunctionSpace(mesh, "DG", 1)

    # Create limiter
    limiter = VertexBasedLimiter(v)

    # advecting velocity
    u0 = conditional(x[0] < 0.5, 1., 0.)
    u = Function(v).interpolate(u0)
    limiter.apply(u)

    assert np.max(u.dat.data_ro) <= 1.0, "Failed by exceeding max values"
    assert np.min(u.dat.data_ro) >= 0.0, "Failed by exceeding min values"


def test_step_function_loop(mesh, iterations=100):
    # test function space
    v = FunctionSpace(mesh, "DG", 1)
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
    D0 = conditional(x[0] < 0.5, 1., 0.)

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


def test_parallel_limiting(tmpdir):
    import pickle
    mesh = RectangleMesh(10, 4, 5000., 1000.)
    V = FunctionSpace(mesh, 'DG', 1)
    f = Function(V)
    x, *_ = SpatialCoordinate(mesh)
    f.project(sin(2*pi*x/3000.))
    limiter = VertexBasedLimiter(V)
    limiter.apply(f)

    expect = np.asarray([norm(f),
                         norm(limiter.centroids),
                         norm(limiter.min_field),
                         norm(limiter.max_field)])

    tmpfile = tmpdir.join("a")
    code = """
import pickle
from firedrake import *
mesh = RectangleMesh(10, 4, 5000., 1000.)
V = FunctionSpace(mesh, 'DG', 1)
f = Function(V)
x, *_ = SpatialCoordinate(mesh)
f.project(sin(2*pi*x/3000.))
limiter = VertexBasedLimiter(V)
limiter.apply(f)

fnorm = norm(f)
centroid_norm = norm(limiter.centroids)
min_norm = norm(limiter.min_field)
max_norm = norm(limiter.max_field)
if mesh.comm.rank == 0:
    with open("{file}", "wb") as f:
        pickle.dump([fnorm, centroid_norm, min_norm, max_norm], f)
""".format(file=tmpfile)
    subprocess.check_call(["mpiexec", "-n", "3", sys.executable, "-c", code])
    with tmpfile.open("rb") as f:
        actual = np.asarray(pickle.load(f))
    assert np.allclose(expect, actual)
