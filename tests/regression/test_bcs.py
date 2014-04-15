import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def V():
    return FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)


@pytest.fixture
def VV():
    return VectorFunctionSpace(UnitSquareMesh(10, 10), "CG", 1)


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture
def a(u, V):
    v = TestFunction(V)
    a = dot(grad(v), grad(u)) * dx
    return a


@pytest.mark.parametrize('v', [0, 1.0])
def test_init_bcs(V, v):
    "Initialise a DirichletBC."
    assert DirichletBC(V, v, 0).function_arg == v


@pytest.mark.parametrize('v', [(0, 0), 'foo'])
def test_init_bcs_illegal(V, v):
    "Initialise a DirichletBC with illegal values."
    with pytest.raises(RuntimeError):
        DirichletBC(V, v, 0)


@pytest.mark.parametrize('v', [[0.0, 0.0], (1.0, 1.0)])
def test_init_vector_bcs(VV, v):
    "Initialise a DirichletBC on a VectorFunctionSpace."
    assert DirichletBC(VV, v, 0).function_arg


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


def test_apply_bcs_wrong_fs(V):
    "Applying a DirichletBC to a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    f = Function(FunctionSpace(V.mesh(), "CG", 2))
    with pytest.raises(RuntimeError):
        bc.apply(f)


def test_zero_bcs_wrong_fs(V):
    "Zeroing a DirichletBC on a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    f = Function(FunctionSpace(V.mesh(), "CG", 2))
    with pytest.raises(RuntimeError):
        bc.zero(f)


def test_init_bcs_wrong_fs(V):
    "Initialise a DirichletBC with a Function on an incompatible FunctionSpace."
    f = Function(FunctionSpace(V.mesh(), "CG", 2))
    with pytest.raises(RuntimeError):
        DirichletBC(V, f, 1)


def test_set_bcs_wrong_fs(V):
    "Set a DirichletBC to a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    f = Function(FunctionSpace(V.mesh(), "CG", 2))
    with pytest.raises(RuntimeError):
        bc.set_value(f)


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


def test_assemble_mass_bcs_2d(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Expression(['x[0]'] * V.dim))

    bcs = [DirichletBC(V, 0.0, 1),
           DirichletBC(V, 1.0, 2)]

    w = Function(V)
    solve(dot(u, v)*dx == dot(f, v)*dx, w, bcs=bcs)

    assert assemble(dot((w - f), (w - f))*dx) < 1e-12


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
