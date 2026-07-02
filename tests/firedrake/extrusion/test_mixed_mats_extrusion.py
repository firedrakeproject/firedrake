import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def m(request):
    return ExtrudedMesh(UnitTriangleMesh(), layers=2, layer_height=0.5)


@pytest.fixture(scope='module')
def V(m):
    return FunctionSpace(m, 'DG', 0)


@pytest.fixture(scope='module')
def Q(m):
    return FunctionSpace(m, 'DG', 0)


@pytest.fixture(scope='module')
def W(V, Q):
    return V*Q


# NOTE: these tests make little to no mathematical sense, they are
# here to exercise corner cases in pyop3's handling of mixed spaces.
def test_massVW0(V, W):
    u = TrialFunction(V)
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert not np.allclose(A.M[label0, :].values, 0.0)
    # DGxRT block (0, since test function was restricted to DG block)
    assert np.allclose(A.M[label1, :].values, 0.0)


def test_massVW1(V, W):
    u = TrialFunction(V)
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block (0, since test function was restricted to RT block)
    assert np.allclose(A.M[label0, :].values, 0.0)
    # DGxRT block
    assert not np.allclose(A.M[label1, :].values, 0.0)


def test_massW0W0(W):
    u = TrialFunction(W)[0]
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert not np.allclose(A.M[label0, label0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[label1, label0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[label0, label1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[label1, label1].values, 0.0)


def test_massW1W1(W):
    u = TrialFunction(W)[1]
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert np.allclose(A.M[label0, label0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[label1, label0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[label0, label1].values, 0.0)
    # RTxRT block
    assert not np.allclose(A.M[label1, label1].values, 0.0)


def test_massW0W1(W):
    u = TrialFunction(W)[0]
    v = TestFunction(W)[1]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert np.allclose(A.M[label0, label0].values, 0.0)
    # DGxRT block
    assert not np.allclose(A.M[label1, label0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[label0, label1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[label1, label1].values, 0.0)


def test_massW1W0(W):
    u = TrialFunction(W)[1]
    v = TestFunction(W)[0]
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert np.allclose(A.M[label0, label0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[label1, label0].values, 0.0)
    # RTxDG block
    assert not np.allclose(A.M[label0, label1].values, 0.0)
    # RTxRT block
    assert np.allclose(A.M[label1, label1].values, 0.0)


def test_massWW(W):
    u = TrialFunction(W)
    v = TestFunction(W)
    A = assemble(inner(u, v)*dx)

    label0, label1 = W._labels
    # DGxDG block
    assert not np.allclose(A.M[label0, label0].values, 0.0)
    # DGxRT block
    assert np.allclose(A.M[label1, label0].values, 0.0)
    # RTxDG block
    assert np.allclose(A.M[label0, label1].values, 0.0)
    # RTxRT block
    assert not np.allclose(A.M[label1, label1].values, 0.0)
