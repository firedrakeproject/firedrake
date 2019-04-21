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

    return u*v*dx


@pytest.fixture(params=["nest", "aij", "matfree"])
def mat_type(request):
    return request.param


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, matrix.Matrix)


def test_assemble_is_lazy(a):
    A = assemble(a)

    assert not A.assembled
    assert A._assembly_callback is not None

    assert (A._M.values == 0.0).all()


def test_M_forces_assemble(a):
    A = assemble(a)

    assert not A.assembled
    assert not (A.M.values == 0.0).all()
    assert A.assembled


def test_solve_forces_assemble(a, V):
    A = assemble(a)

    v = TestFunction(V)
    f = Function(V)

    b = assemble(f*v*dx)
    assert not A.assembled

    solve(A, f, b)
    assert A.assembled
    assert not (A._M.values == 0.0).all()


def test_assemble_with_bcs_then_not(a, V):
    bc1 = DirichletBC(V, 0, 1)
    A = assemble(a, bcs=[bc1])
    Abcs = A.M.values

    A = assemble(a)
    assert not A.has_bcs
    Anobcs = A.M.values

    assert (Anobcs != Abcs).any()

    A = assemble(a, bcs=[bc1])
    Abcs = A.M.values
    assemble(a, tensor=A)
    Anobcs = A.M.values
    assert not A.has_bcs
    assert (Anobcs != Abcs).any()


def test_assemble_with_bcs_multiple_subdomains(a, V, mat_type):
    bc1 = DirichletBC(V, 0, [1, 2])
    A = assemble(a, bcs=[bc1], mat_type=mat_type)
    assert not A.assembled
    assert A._needs_reassembly
    A.assemble()
    assert A.assembled
    assert not A._needs_reassembly
