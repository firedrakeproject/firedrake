import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def V():
    return FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture
def a(u, V):
    v = TestFunction(V)
    a = dot(grad(v), grad(u)) * dx
    return a


@pytest.mark.parametrize('measure', [dx, ds])
def test_assemble_bcs_wrong_fs(V, measure):
    "Assemble a Matrix with a DirichletBC on an incompatible FunctionSpace."
    u, v = TestFunction(V), TrialFunction(V)
    W = FunctionSpace(V.mesh(), "CG", 2)
    A = assemble(u*v*measure, bcs=[DirichletBC(W, 32, 1)])
    with pytest.raises(RuntimeError):
        A.M.values


def test_assemble_bcs_wrong_fs_interior(V):
    "Assemble a Matrix with a DirichletBC on an incompatible FunctionSpace."
    u, v = TestFunction(V), TrialFunction(V)
    W = FunctionSpace(V.mesh(), "CG", 2)
    n = FacetNormal(V.mesh())
    A = assemble(inner(jump(u, n), jump(v, n))*dS, bcs=[DirichletBC(W, 32, 1)])
    with pytest.raises(RuntimeError):
        A.M.values


def test_homogeneous_bcs(a, u, V):
    bcs = [DirichletBC(V, 32, 1)]

    [bc.homogenize() for bc in bcs]
    # Compute solution - this should have the solution u = 0
    solve(a == 0, u, bcs=bcs)

    assert max(abs(u.vector().array())) == 0.0


def test_homogenize_doesnt_overwrite_function(a, u, V):
    f = Function(V)
    f.assign(10)
    bc = DirichletBC(V, f, 1)
    bc.homogenize()

    assert (f.vector().array() == 10.0).all()

    solve(a == 0, u, bcs=[bc])
    assert max(abs(u.vector().array())) == 0.0


def test_restore_bc_value(a, u, V):
    f = Function(V)
    f.assign(10)
    bc = DirichletBC(V, f, 1)
    bc.homogenize()

    solve(a == 0, u, bcs=[bc])
    assert max(abs(u.vector().array())) == 0.0

    bc.restore()
    solve(a == 0, u, bcs=[bc])
    assert np.allclose(u.vector().array(), 10.0)


def test_set_bc_value(a, u, V):
    f = Function(V)
    f.assign(10)
    bc = DirichletBC(V, f, 1)

    bc.set_value(7)

    solve(a == 0, u, bcs=[bc])

    assert np.allclose(u.vector().array(), 7.0)


def test_preassembly_change_bcs(V):
    v = TestFunction(V)
    u = TrialFunction(V)
    a = u*v*dx
    f = Function(V)
    f.assign(10)
    bc = DirichletBC(V, f, 1)

    A = assemble(a, bcs=[bc])
    L = v*f*dx
    b = assemble(L)

    y = Function(V)
    y.assign(7)
    bc1 = DirichletBC(V, y, 1)
    u = Function(V)

    solve(A, u, b)
    assert np.allclose(u.vector().array(), 10.0)

    u.assign(0)
    b = assemble(v*y*dx)
    solve(A, u, b, bcs=[bc1])
    assert np.allclose(u.vector().array(), 7.0)


def test_preassembly_doesnt_modify_assembled_rhs(V):
    v = TestFunction(V)
    u = TrialFunction(V)
    a = u*v*dx
    f = Function(V)
    f.assign(10)
    bc = DirichletBC(V, f, 1)

    A = assemble(a, bcs=[bc])
    L = v*f*dx
    b = assemble(L)

    b_vals = b.vector().array()

    u = Function(V)
    solve(A, u, b)
    assert np.allclose(u.vector().array(), 10.0)

    assert np.allclose(b_vals, b.vector().array())


def test_preassembly_bcs_caching(V):
    bc1 = DirichletBC(V, 0, 1)
    bc2 = DirichletBC(V, 1, 2)

    v = TestFunction(V)
    u = TrialFunction(V)

    a = u*v*dx

    Aboth = assemble(a, bcs=[bc1, bc2])
    Aneither = assemble(a)
    A1 = assemble(a, bcs=[bc1])
    A2 = assemble(a, bcs=[bc2])

    assert not np.allclose(Aboth.M.values, Aneither.M.values)
    assert not np.allclose(Aboth.M.values, A2.M.values)
    assert not np.allclose(Aboth.M.values, A1.M.values)
    assert not np.allclose(Aneither.M.values, A2.M.values)
    assert not np.allclose(Aneither.M.values, A1.M.values)
    assert not np.allclose(A2.M.values, A1.M.values)
    # There should be no zeros on the diagonal
    assert not any(A2.M.values.diagonal() == 0)
    assert not any(A1.M.values.diagonal() == 0)
    assert not any(Aneither.M.values.diagonal() == 0)


def test_mass_bcs_1d():
    m = UnitIntervalMesh(5)
    V = FunctionSpace(m, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.interpolate(Expression('x[0]'))

    bcs = [DirichletBC(V, 0.0, 1),
           DirichletBC(V, 1.0, 2)]

    w = Function(V)
    solve(u*v*dx == f*v*dx, w, bcs=bcs)

    assert assemble((w - f)*(w - f)*dx) < 1e-10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
