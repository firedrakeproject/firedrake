from firedrake import *
from firedrake import types
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

    assert isinstance(A, types.Matrix)


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


def test_adding_bcs(a, V):
    bc1 = DirichletBC(V, 0, 1)
    A = assemble(a, bcs=[bc1])

    for bc in A.bcs:
        assert bc is bc1

    bc2 = DirichletBC(V, 1, 1)
    bc2.apply(A)
    for bc in A.bcs:
        assert bc is bc2

    bc3 = DirichletBC(V, 1, 0)

    bc3.apply(A)
    assert A.bcs == set([bc2, bc3])


def test_assemble_with_bcs(a, V):
    bc1 = DirichletBC(V, 0, 1)
    A = assemble(a, bcs=[bc1])

    A.assemble()
    assert A.assembled
    assert not A._needs_reassembly

    # Same subdomain, should not need reassembly
    bc2 = DirichletBC(V, 1, 1)
    bc2.apply(A)
    assert A.assembled
    assert not A._needs_reassembly

    bc3 = DirichletBC(V, 1, 0)
    A.bcs = bc3
    assert A.assembled
    assert A._needs_reassembly
    A.assemble()
    assert A.assembled
    assert not A._needs_reassembly

    bc2.apply(A)
    assert A.assembled
    assert A._needs_reassembly
    A.assemble()
    assert A.assembled
    assert not A._needs_reassembly


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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
