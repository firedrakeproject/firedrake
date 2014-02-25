from firedrake import *
import pytest


@pytest.fixture(scope='module')
def V():
    m = UnitSquareMesh(25, 25)
    return FunctionSpace(m, 'CG', 1)


def test_nullspace(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = -v*ds(3) + v*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    solve(a == L, u, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(Expression('x[1] - 0.5'))
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 5e-8


def test_nullspace_preassembled(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = -v*ds(3) + v*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    A = assemble(a)
    b = assemble(L)
    solve(A, u, b, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(Expression('x[1] - 0.5'))
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 5e-8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
