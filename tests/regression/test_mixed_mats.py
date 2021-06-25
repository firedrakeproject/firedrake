import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def m(request):
    return UnitTriangleMesh()


@pytest.fixture(scope='module')
def V(m):
    return FunctionSpace(m, 'DG', 0)


@pytest.fixture(scope='module')
def Q(m):
    return FunctionSpace(m, 'RT', 1)


@pytest.fixture(scope='module')
def W(V, Q):
    return V*Q


# NOTE: these tests make little to no mathematical sense, they are
# here to exercise corner cases in PyOP2's handling of mixed spaces.
def test_massVW0(V, W):
    u = TrialFunction(V)
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 1)
    # DGxDG block
    assert not np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block (0, since test function was restricted to DG block)
    assert np.allclose(A.M[1, 0].values, 0.0)


def test_massVW1(V, W):
    u = TrialFunction(V)
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 1)
    # DGxDG block (0, since test function was restricted to RT block)
    assert np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert not np.allclose(A.M[1, 0].values, 0.0)


def test_massW0W0(W):
    u = TrialFunction(W)[0]
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 2)
    # DGxDG block
    assert not np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[1, 0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[0, 1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[1, 1].values, 0.0)


def test_massW1W1(W):
    u = TrialFunction(W)[1]
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 2)
    # DGxDG block
    assert np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[1, 0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[0, 1].values, 0.0)
    # RTxRT block
    assert not np.allclose(A.M[1, 1].values, 0.0)


def test_massW0W1(W):
    u = TrialFunction(W)[0]
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 2)
    # DGxDG block
    assert np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert not np.allclose(A.M[1, 0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[0, 1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[1, 1].values, 0.0)


def test_massW1W0(W):
    u = TrialFunction(W)[1]
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 2)
    # DGxDG block
    assert np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[1, 0].values, 0.0)
    # RTxDG block
    assert not np.allclose(A.M[0, 1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[1, 1].values, 0.0)


def test_massWW(W):
    u = TrialFunction(W)
    v = TestFunction(W)
    A = assemble(inner(u, v)*dx)
    assert A.M.sparsity.shape == (2, 2)
    # DGxDG block
    assert not np.allclose(A.M[0, 0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[1, 0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[0, 1].values, 0.0)
    # RTxRT block
    assert not np.allclose(A.M[1, 1].values, 0.0)


def test_bcs_ordering():
    """Check that application of boundary conditions zeros the correct
    rows and columns of a mixed matrix.

    The diagonal blocks should get a 1 in the diagonal entries
    corresponding to the boundary condition nodes, the corresponding
    rows and columns in the whole system should be zeroed."""
    m = UnitIntervalMesh(5)
    V = FunctionSpace(m, 'CG', 1)
    W = V*V
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    bc1 = DirichletBC(W.sub(0), 0, 1)
    bc2 = DirichletBC(W.sub(1), 1, 2)

    a = (inner(u, v) + inner(u, q) + inner(p, v) + inner(p, q))*dx

    A = assemble(a, bcs=[bc1, bc2])

    assert np.allclose(A.M[0, 0].values.diagonal()[bc1.nodes], 1.0)
    assert np.allclose(A.M[1, 1].values.diagonal()[bc2.nodes], 1.0)
    assert np.allclose(A.M[0, 1].values[bc1.nodes, :], 0.0)
    assert np.allclose(A.M[1, 0].values[:, bc1.nodes], 0.0)
    assert np.allclose(A.M[1, 0].values[bc2.nodes, :], 0.0)
    assert np.allclose(A.M[0, 1].values[:, bc2.nodes], 0.0)
