import pytest

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
    # DGxDG block (0, since test function was restricted to DG block)
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
