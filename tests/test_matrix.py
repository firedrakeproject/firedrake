from firedrake import *
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


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, Matrix)


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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
