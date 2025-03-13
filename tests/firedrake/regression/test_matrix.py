from firedrake import *
from firedrake import matrix
import pytest


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def test(V):
    return TestFunction(V)


@pytest.fixture
def trial(V):
    return TrialFunction(V)


@pytest.fixture
def a(test, trial):
    return inner(trial, test) * dx


@pytest.fixture(params=["nest", "aij", "matfree"])
def mat_type(request):
    return request.param


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, matrix.Matrix)


def test_solve_with_assembled_matrix(a):
    (v, u) = a.arguments()
    V = v.function_space()
    x, = SpatialCoordinate(V.mesh())
    f = Function(V).interpolate(x)

    A = AssembledMatrix((v, u), bcs=(), petscmat=assemble(a).petscmat)
    L = inner(f, v) * dx

    solution = Function(V)
    solve(A == L, solution)

    assert norm(assemble(f - solution)) < 1e-15
