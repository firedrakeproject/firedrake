from firedrake import *
from firedrake import matrix
import pytest


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def a(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    return inner(u, v)*dx


@pytest.fixture(params=["nest", "aij", "matfree"])
def mat_type(request):
    return request.param


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, matrix.Matrix)
